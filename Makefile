sync_up:
	rsync --rsync-path="sudo rsync" -av --chown 1001:1001 *.py erpipehe@jyu2401-62.ws:/var/data/open-webui-setup/jupyterhub_home/ihmistieteilija/notebooks

sync_down:
	rsync --rsync-path="sudo rsync" -av --chown 1001:1001 erpipehe@jyu2401-62.ws:/var/data/open-webui-setup/jupyterhub_home/ihmistieteilija/notebooks/*.py .
