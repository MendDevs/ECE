
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

 python -m venv .venv    

.venv\Scripts\activate.ps1  || source .venv/bin/activate  (Mac)

# Renseigner tout les packages que vous allez utiliser dans le fichier requirements

pip install -r requirenments.txt


