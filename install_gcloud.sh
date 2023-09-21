apt-get update
apt-get install apt-transport-https ca-certificates gnupg curl sudo
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.asc
apt-get update && apt-get install google-cloud-cli

# then you need to do:
gcloud auth login # user authentication
gcloud auth application-default login # SDK/CLI authentication
gcloud config set project ml-dev-a7b7 # set default project
gcloud auth application-default set-quota-project ml-dev-a7b7 # swt default quota project
