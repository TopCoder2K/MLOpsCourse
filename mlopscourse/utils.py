import subprocess


# Credits to https://stackoverflow.com/a/21901260/12187881
def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
