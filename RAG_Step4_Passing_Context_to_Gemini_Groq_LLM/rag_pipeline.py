import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from google import genai
from google.genai import types

load_dotenv()

DOCUMENTS = {
    "git_basics.txt": """Git Basics

Git is a version control system that records the history of a project as a series of snapshots
called commits. Each commit stores what changed, who made the change, and a message that explains
why. This lets developers go back to an earlier version at any time.

A branch is a separate line of development. Developers create a branch to try a new feature without
affecting the main code, and later merge it back when the work is ready.

A remote is a copy of the repository hosted online, such as on GitHub. The command git push uploads
local commits to the remote, and git pull downloads new commits from it. If the remote contains
commits you do not have, Git rejects your push and asks you to pull first.

A file named .gitignore lists files that Git should never track, such as secret keys, virtual
environments, and temporary files. Adding .env to .gitignore stops passwords and API keys from
being uploaded to GitHub by accident.
""",
    "linux_commands.txt": """Linux Command Line Essentials

The command line lets you control a Linux system by typing commands. The command ls lists the files
in a folder, cd changes the current folder, and pwd prints the full path of the folder you are in.

The command grep searches inside files for text that matches a pattern. The command find searches
for files by name, size, or date. Both are used constantly to locate things in large projects.

Every Linux file has permissions that say who can read, write, or execute it. The command chmod
changes these permissions. Be careful with commands such as rm -rf, which deletes files permanently
without asking, because there is no recycle bin on the command line.
""",
    "python_environments.txt": """Python Environments and Secrets

A virtual environment is an isolated folder that holds the Python packages for one project. Using a
separate environment for each project prevents version conflicts, because two projects can depend
on different versions of the same library.

The command pip install downloads packages into the active environment. A file called
requirements.txt lists the packages a project needs, so another person can install them all with a
single command.

Secrets such as API keys should never be written directly in the code. Instead they are stored in a
file named .env, and the program reads them at run time using a library like python-dotenv. The .env
file must be kept out of version control so the keys are not shared.
""",
}

NO_ANSWER = "I don't know based on the provided documents."

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    f"If the answer is not in the context, reply exactly: \"{NO_ANSWER}\" "
    "Keep the answer short and clear. At the end, mention the source file name(s) you used."
)


class LLMClient:
    """Calls Groq (or Gemini if there is no Groq key). Raises an error if every model fails."""

    GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
    GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]

    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not (self.groq_key or self.gemini_key):
            raise SystemExit("No API key found. Put GROQ_API_KEY=... in a .env file next to this script.")
        self.provider = "groq" if self.groq_key else "gemini"

    def generate(self, user_prompt, system_prompt, temperature=0.1):
        errors = []

        if self.provider == "groq":
            client = Groq(api_key=self.groq_key)
            for model in self.GROQ_MODELS:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_prompt}],
                        temperature=temperature,
                        max_tokens=600,
                    )
                    text = response.choices[0].message.content
                    if text:
                        return text.strip()
                except Exception as e:
                    errors.append(f"{model}: {e}")
        else:
            client = genai.Client(api_key=self.gemini_key)
            for model in self.GEMINI_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                            max_output_tokens=1500,
                        ),
                    )
                    if response.text:
                        return response.text.strip()
                except Exception as e:
                    errors.append(f"{model}: {e}")

        raise RuntimeError("All models failed -> " + " | ".join(errors))


class RAGPipeline:
    def __init__(self, documents=None, chunk_size=250, chunk_overlap=40, threshold=0.25):
        self.threshold = threshold
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = chromadb.Client().get_or_create_collection(
            name="rag_chunks", embedding_function=embedder
        )
        self.llm = LLMClient()
        print("Using provider:", self.llm.provider)

        for source, text in (documents or DOCUMENTS).items():
            self.add_document(source, text)
        print("Total chunks stored:", self.collection.count())

    def add_document(self, source, text):
        chunks = self.splitter.split_text(text)
        self.collection.upsert(          # upsert = safe to run twice, no duplicate-id errors
            documents=chunks,
            metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
            ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
        )
        return len(chunks)

    def retrieve(self, query, n_results=4):
        n = min(n_results, self.collection.count())
        results = self.collection.query(
            query_texts=[query], n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = 1 - dist
            if similarity >= self.threshold:
                chunks.append({"text": doc, "source": meta["source"],
                               "chunk_index": meta["chunk_index"], "similarity": round(similarity, 4)})
        return chunks

    @staticmethod
    def build_prompt(query, chunks):
        context = "\n\n".join(
            f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
        )
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    def ask(self, query, n_results=4, system_prompt=None, temperature=0.1):
        """Returns (answer_text, chunks_used). Skips the LLM completely if nothing relevant is found."""
        chunks = self.retrieve(query, n_results=n_results)
        if not chunks:
            return NO_ANSWER, []
        answer = self.llm.generate(
            self.build_prompt(query, chunks),
            system_prompt or self.system_prompt,
            temperature=temperature,
        )
        return answer, chunks