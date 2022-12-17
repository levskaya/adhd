Create two terminal windows.

### Go to a one terminal window to create the tmux session we'll be using for remote access
```bash
tmux new-session -s pod_control_pane
```

### On another terminal window

Create the pod slice.

```bash
gcloud compute tpus tpu-vm create my_tpu_name --zone us-central1-a --project my-project --accelerator-type v2-32 --version tpu-vm-base
```

Be sure to include GCE keys.

```bash
ssh-add ~/.ssh/google_compute_engine
```

Run setup / tmux layout script.

```bash
python cloud/setup_hosts.py --tpu_name=my_tpu_name --zone us-central1-a --project my-project
```

If you desynchronize files with rsync and cause a hang across hosts, kill them using the following helper.

```bash
python cloud/kill_hanging_processes.py  --tpu_name=my_tpu_name --zone us-central1-a --project my-project --proc_name=train.py
```

Kill podslice when done.

```bash
gcloud compute tpus tpu-vm delete my_tpu_name --zone us-central1-a --project my-project
```
