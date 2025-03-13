# DEEP-LEARNING-AY-2024-2025

## 1️⃣ Navigate to Your Project Directory
If you’re not already inside your project folder, navigate there:

`cd your-project-folder`

## 3️⃣ Start the Container
If your container was previously built, just start it:

`docker run -it dl_container`  

`docker exec -it dl_container bash`
Starts in detached mode (background)
If you didn’t build the image before, or you made changes, rebuild and run it:

`docker build -t dl_container .`

`docker run -it dl_container`

## 4️⃣ Stop the Container (Optional)
When you're done, you can stop the running containers with:

`exit()`

This ensures your ML environment is ready to use every time you open your Codespace.

## 5️⃣ Commit Changes (Optional)
If you want to save the state of the container (including installed packages and changes to the filesystem), you can commit the container to a new image:

`docker commit dl_container dl_container_updated`