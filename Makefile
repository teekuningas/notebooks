JYU_SERVER = erpipehe@jyu2401-62.ws
BIRD_SERVER = root@130.234.36.115

sync_jyu2401_62_up:
	rsync --rsync-path="sudo rsync" -av --chown 1001:1001 *.py $(JYU_SERVER):/var/data/open-webui-setup/jupyterhub_home/ihmistieteilija/notebooks

sync_jyu2401_62_down:
	rsync --rsync-path="sudo rsync" -av $(JYU_SERVER):/var/data/open-webui-setup/jupyterhub_home/ihmistieteilija/notebooks/*.py .

sync_bird_up:
	ssh $(BIRD_SERVER) "mkdir -p /home/user/notebooks && chown user:users /home/user/notebooks"
	rsync -av --chown user:users geo *.py $(BIRD_SERVER):/home/user/notebooks

sync_bird_down:
	rsync -av $(BIRD_SERVER):/home/user/notebooks/*.py .
