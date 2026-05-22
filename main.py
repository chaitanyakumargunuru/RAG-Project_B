import os
import shutil
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt

# ── IMPORT MEMBER C'S RAG CORE FUNCTIONS ──
from rag.loader import load_document
from rag.splitter import split_documents
from rag.vectorstore import add_documents
from rag.retriever import retrieve_and_answer
from config import UPLOAD_DIR

# ==========================================
# 1. CONFIGURATION & DATABASE ACCESS SETUP
# ==========================================
SECRET_KEY = "super_secret_change_this_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# PostgreSQL setup with safe character URL encoding
DB_USER = "postgres"
DB_PASSWORD = "C-pad@21"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "rag_db"

safe_password = urllib.parse.quote_plus(DB_PASSWORD)
DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="Unified Secured RAG Application Backend Layer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. DATA ORM MODELLING & SCHEMAS
# ==========================================
class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class UserSignUp(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: EmailStr

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class ChatInput(BaseModel):
    question: str


# ==========================================
# 3. UTILITY CORE CRYPTOGRAPHY SECURITY
# ==========================================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate active credentials context",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(DBUser).filter(DBUser.email == email).first()
    if user is None:
        raise credentials_exception
    return user


# ==========================================
# 4. API ROUTERS & ENDPOINTS
# ==========================================

@app.post("/api/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user_data: UserSignUp, db: Session = Depends(get_db)):
    existing_user = db.query(DBUser).filter(DBUser.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email address is already registered.")

    new_user = DBUser(
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        email=user_data.email,
        hashed_password=hash_password(user_data.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/api/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email authentication parameters or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/upload")
async def upload_document(
        file: UploadFile = File(...),
        current_user: DBUser = Depends(get_current_user)
):
    """
    Protected Resource: Receives document file, performs user multi-tenant isolation,
    and runs Member C's full ingestion/vectorization pipeline block.
    """
    # Isolate user contexts by string casting their unique integer database index ID
    user_tenant_id = str(current_user.id)

    # Verify processing compatibility criteria types
    allowed_ext = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document format choice '{ext}'. Allowed extensions: {', '.join(allowed_ext)}"
        )

    # Construct distinct ingestion path tracing boundary
    safe_filename = f"user_{user_tenant_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # EXECUTE MEMBER C'S PIPELINE STACKS MECHANICS
        documents = load_document(file_path)  # Extractor Block
        chunks = split_documents(documents)  # Text Overlap Block
        chunks_vectorized_count = add_documents(user_tenant_id, chunks)  # Storage Ingestion ChromaDB
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"System breakdown during vectorization layout execution processing: {str(e)}"
        )

    return {
        "filename": file.filename,
        "status": "Indexed inside individual secure user text pool vector collection successfully.",
        "chunks_stored": chunks_vectorized_count
    }


@app.post("/api/chat")
def chat_with_documents(
        payload: ChatInput,
        current_user: DBUser = Depends(get_current_user)
):
    """
    Protected Resource: Accepts query string, retrieves relevant contextual vectors
    belonging ONLY to this specific user, and queries the local LLM for answers.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Inquiry message content field cannot be sent empty.")

    user_tenant_id = str(current_user.id)

    try:
        # EXECUTE MEMBER C'S RETRIEVE AND GENERATE ENGINE
        rag_engine_output = retrieve_and_answer(
            user_id=user_tenant_id,
            question=payload.question
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Q&A loop failure inside the local LLM context connector: {str(e)}. Make sure Ollama is open."
        )

    return {
        "answer": rag_engine_output["answer"],
        "sources": rag_engine_output["sources"]
    }