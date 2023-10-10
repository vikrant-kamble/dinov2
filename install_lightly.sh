pip install virtualenv
pushd /root
virtualenv lightly
source lightly/bin/activate
pip install lightly ipykernel
python -m ipykernel install --name=lightly
popd
