.PHONY: env
env:
	mamba env create -f environment.yml -p ~/envs/ligo
	bash -ic 'conda activate ligo;python -m ipykernel install --user --name ligo --display-name "li

html:
	jupyter-book build .
	
conf.py: _config.yml _toc.yml
	jupyter-book config sphinx .

html-hub: conf.py
	jupyter-book config sphinx .
	sphinx-build  . _build/html -D html_baseurl=${JUPYTERHUB_SERVICE_PREFIX}/proxy/absolute/8000
	@echo "Start the Python http server in another terminal and visit:"
	@echo "https://stat159.datahub.berkeley.edu/user-redirect/proxy/8000/index.html"

.PHONY: clean
clean:
	rm -rf _build/html/
	rm -f figures/*.png
	rm -f data/*.csv
	rm -f audio/*.wav
	
	