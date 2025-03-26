import os
from dotenv import load_dotenv
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from office365.runtime.auth.user_credential import UserCredential

load_dotenv()  # Charge les variables depuis le fichier .env

class SharePointDownloader:
    def __init__(self):
        self.site_url = os.getenv("SHAREPOINT_SITE_URL")
        self.username = os.getenv("SHAREPOINT_USERNAME")
        self.password = os.getenv("SHAREPOINT_PASSWORD")
        self.folder_url = os.getenv("SHAREPOINT_FOLDER_URL")
        self.local_path = os.getenv("LOCAL_DOWNLOAD_PATH", "sharepoint_downloads")
        self.ctx = None

    def connect(self):
        """Établir une connexion à SharePoint"""
        try:
            self.ctx = ClientContext(self.site_url).with_credentials(
                UserCredential(self.username, self.password)
            )
            web = self.ctx.web
            self.ctx.load(web)
            self.ctx.execute_query()
            print(f"Connexion réussie à {web.url}")
            return True
        except Exception as e:
            print(f"Échec de la connexion : {str(e)}")
            return False

    def download_folder(self):
        """Télécharger le contenu d'un dossier SharePoint"""
        try:
            folder = self.ctx.web.get_folder_by_server_relative_url(self.folder_url)
            self.ctx.load(folder)
            self.ctx.execute_query()

            # Télécharger les fichiers
            files = folder.files
            self.ctx.load(files)
            self.ctx.execute_query()

            os.makedirs(self.local_path, exist_ok=True)
            for file in files:
                try:
                    file_content = File.open_binary(self.ctx, file.serverRelativeUrl)
                    local_file_path = os.path.join(self.local_path, file.name)
                    with open(local_file_path, "wb") as f:
                        f.write(file_content.content)
                    print(f"Téléchargé : {file.name}")
                except Exception as e:
                    print(f"Échec du téléchargement de {file.name} : {str(e)}")

            # Télécharger les sous-dossiers récursivement
            subfolders = folder.folders
            self.ctx.load(subfolders)
            self.ctx.execute_query()

            for subfolder in subfolders:
                subfolder_local_path = os.path.join(self.local_path, subfolder.name)
                self.download_subfolder(subfolder, subfolder_local_path)

            return True
        except Exception as e:
            print(f"Échec du téléchargement du dossier : {str(e)}")
            return False

    def download_subfolder(self, folder, local_path):
        """Télécharger récursivement un sous-dossier"""
        try:
            self.ctx.load(folder)
            self.ctx.execute_query()

            files = folder.files
            self.ctx.load(files)
            self.ctx.execute_query()

            os.makedirs(local_path, exist_ok=True)
            for file in files:
                try:
                    file_content = File.open_binary(self.ctx, file.serverRelativeUrl)
                    local_file_path = os.path.join(local_path, file.name)
                    with open(local_file_path, "wb") as f:
                        f.write(file_content.content)
                    print(f"Téléchargé : {file.name}")
                except Exception as e:
                    print(f"Échec du téléchargement de {file.name} : {str(e)}")

            subfolders = folder.folders
            self.ctx.load(subfolders)
            self.ctx.execute_query()

            for subfolder in subfolders:
                subfolder_local_path = os.path.join(local_path, subfolder.name)
                self.download_subfolder(subfolder, subfolder_local_path)

        except Exception as e:
            print(f"Échec du téléchargement du sous-dossier : {str(e)}")

def main():
    downloader = SharePointDownloader()

    # Vérifier que les paramètres essentiels sont présents
    if not all([downloader.site_url, downloader.username, downloader.password, downloader.folder_url]):
        print("Erreur : Les paramètres essentiels (site_url, username, password, folder_url) doivent être configurés dans le fichier .env")
        return

    if downloader.connect():
        success = downloader.download_folder()
        if success:
            print("Téléchargement terminé avec succès")
        else:
            print("Téléchargement échoué")

if __name__ == "__main__":
    main()