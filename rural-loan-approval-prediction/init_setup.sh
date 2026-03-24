echo [$(date)]: "START"
echo [$(date)]: "creating conda env with python 3.8 version" 
conda create -n loan python==3.8 -y
echo [$(date)]: "activating the environment" 
conda activate loan
echo [$(date)]: "installing the requirements" 
pip install -r requirements.txt
echo [$(date)]: "END" 
