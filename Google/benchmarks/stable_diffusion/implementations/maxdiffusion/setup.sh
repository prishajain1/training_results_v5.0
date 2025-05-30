TPU_TYPE=v6e-256
ZONE=southamerica-west1-a
REGION=southamerica-west1
CLUSTER_NAME="mlperf-${TPU_TYPE}-${ZONE}"
PROJECT=some-cloud-tpu-project-id
NUM_SLICES=2

NETWORK_NAME_1=${CLUSTER_NAME}-mtu9k-1-${ZONE}
SUBNET_NAME_1=${CLUSTER_NAME}-privatesubnet-1-${ZONE}
NETWORK_FW_NAME_1=${NETWORK_NAME_1}-fw-1-${ZONE}
FIREWALL_RULE_NAME=${CLUSTER_NAME}-privatefirewall-1-${ZONE}
ROUTER_NAME=${CLUSTER_NAME}-network-1-${ZONE}
NAT_CONFIG=${CLUSTER_NAME}-natconfig-1-${ZONE}

# Use a custom network for better performance as well as avoid the default network to be overloaded.
gcloud compute networks create "${NETWORK_NAME_1}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom --project=$PROJECT
gcloud compute networks subnets create "${SUBNET_NAME_1}" --network="${NETWORK_NAME_1}" --range=10.11.0.0/18 --region="${REGION}" --project=$PROJECT
gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" --network "${NETWORK_NAME_1}" --allow tcp,icmp,udp --project="${PROJECT}"
gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT}" \
  --network="${NETWORK_NAME_1}" \
  --region="${REGION}"
gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --project="${PROJECT}" \
  --enable-logging


# Secondary subnet for multi-nic experience. Need custom ip routing to be different from the first network’s subnet.
export NETWORK_NAME_2=${CLUSTER_NAME}-privatenetwork-2-${ZONE}
export SUBNET_NAME_2=${CLUSTER_NAME}-privatesubnet-2-${ZONE}
export FIREWALL_RULE_NAME=${CLUSTER_NAME}-privatefirewall-2-${ZONE}
export ROUTER_NAME=${CLUSTER_NAME}-network-2-${ZONE}
export NAT_CONFIG=${CLUSTER_NAME}-natconfig-2-${ZONE}

gcloud compute networks create "${NETWORK_NAME_2}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom --project=$PROJECT
gcloud compute networks subnets create "${SUBNET_NAME_2}" --network="${NETWORK_NAME_2}" --range=10.10.0.0/18 --region="${REGION}" --project=$PROJECT
gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" --network "${NETWORK_NAME_2}" --allow tcp,icmp,udp --project="${PROJECT}"
gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT}" \
  --network="${NETWORK_NAME_2}" \
  --region="${REGION}"
gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --project="${PROJECT}" \
  --enable-logging

export CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias --enable-multi-networking --network=${NETWORK_NAME_1} --subnetwork=${SUBNET_NAME_1}"

export NODE_POOL_ARGUMENTS="--additional-node-network network=${NETWORK_NAME_2},subnetwork=${SUBNET_NAME_2}"

python3 ~/xpk/xpk.py cluster create --cluster $CLUSTER_NAME --num-slices=$NUM_SLICES --tpu-type=$TPU_TYPE --zone=$ZONE  --project=$PROJECT --on-demand --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" --custom-nodepool-arguments="${NODE_POOL_ARGUMENTS}"
