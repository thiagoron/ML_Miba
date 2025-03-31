import os
import subprocess

def install_offline_packages(requirements_file, packages_dir):
    try:
        subprocess.check_call([
            "pip", "install", "--no-index", f"--find-links={packages_dir}", "-r", requirements_file
        ])
        print("Pacotes instalados com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao instalar pacotes: {e}")

if __name__ == "__main__":
    print("Certifique-se de que o Python está instalado antes de continuar.")
    requirements_file = "requirements.txt"
    packages_dir = "./offline_packages"
    
    if os.path.exists(requirements_file) and os.path.exists(packages_dir):
        install_offline_packages(requirements_file, packages_dir)
    else:
        print("Certifique-se de que 'requirements.txt' e a pasta 'offline_packages' estão no mesmo diretório.")