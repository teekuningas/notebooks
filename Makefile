sync:
	rsync --rsync-path="sudo rsync" -av --chown 1001:1001 *.ipynb *.py jyu2401-62:/var/data/open-webui-setup/jupyterhub_home/ihmistieteilija/notebooks
