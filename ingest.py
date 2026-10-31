"""Index your profile (CV, papers, notes in profile/). Re-run whenever those files change."""
from jobagent import config
from jobagent.rag import ProfileIndex

if __name__ == "__main__":
    n = ProfileIndex().build(config.PROFILE_DIR)
    if n:
        print(f"Indexed {n} chunks from '{config.PROFILE_DIR}/'.")
    else:
        print(f"No text found in '{config.PROFILE_DIR}/'. Add .pdf/.md/.txt files and run again.")
