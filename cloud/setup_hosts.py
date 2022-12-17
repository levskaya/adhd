import os
import argparse
import subprocess
import requests

OKGREEN = '\033[92m'
ENDC = '\033[0m'

parser = argparse.ArgumentParser(description='Setup slice workflow.')
parser.add_argument('--tpu_name')
parser.add_argument('--zone')
parser.add_argument('--project')
parser.add_argument("--skip_ssh_connection", default=False, action="store_true",
                    help="skip the setup steps, e.g. if a later step fails")
parser.add_argument("--skip_setup_fswatch", default=False, action="store_true",
                    help="skip the setup steps, e.g. if a later step fails")
# parser.add_argument("--working_dir", default='working_dir/',
#                     help="dir to sync with fswatch")


def get_bearer():
  return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


def tpu_info(name, project, zone):
  headers = {'Authorization': f'Bearer {get_bearer()}'}

  response = requests.get(
      f"https://tpu.googleapis.com/v2alpha1/projects/{project}/locations/{zone}/nodes/{name}",
      headers=headers)

  return response.json()


def tmux(command):
  return os.system('tmux %s' % command)


def tmux_shell(command):
  tmux('send-keys "%s" "C-m"' % command)


def tmux_select_pane(idx):
  tmux(f'select-pane -t {idx}')


if __name__ == "__main__":
  args = parser.parse_args()

  # ---------------------------------------------------------------------------
  # Get worker info and upload repo and setup scripts
  # ---------------------------------------------------------------------------

  if tmux("has-session -t pod_control_pane") != 0:
    raise ValueError("Need a tmux session called 'pod_control_pane' to be running.")

  # Attach to the window titled pod_control_pane in the background
  # NOTE: this is incorrect CLI syntax and can't be doing anything...
  # doing the intended command with -d -t is not what you want to do either.
  tmux("attach -t -d pod_control_pane")

  # Get information about the tpu - specifically the # of hosts and their IP addresses
  info = tpu_info(args.tpu_name, args.project, args.zone)
  num_hosts = len(info['networkEndpoints'])
  print("TPU name ", args.tpu_name)
  print(f"Total hosts: {num_hosts}")
  for idx, addr in enumerate(info['networkEndpoints']):
    print(
        f"Host {idx}: Internal IP: {addr['ipAddress']} External IP {addr['accessConfig']['externalIp']}")
  internal_ips = [addr['ipAddress'] for addr in info['networkEndpoints']]
  external_ips = [addr['accessConfig']['externalIp'] for addr in info['networkEndpoints']]

  adhd_path = os.path.dirname(os.path.dirname(__file__))
  print(f"Uploading and syncing {adhd_path}")

  # Upload adhd directory for the first time
  setup_script_path = os.path.join(os.path.dirname(__file__), "setup.sh")
  print(f"Uploading setup script at {setup_script_path}")
  tmux_shell(f"for ip in {' '.join(external_ips)}; do rsync -ravz --stats --exclude=.git {adhd_path} $ip:~/adhd; done")

  # Upload setup script
  output = subprocess.run(
      f'gcloud compute tpus tpu-vm scp {setup_script_path} {args.tpu_name}:setup.sh --worker=all --zone={args.zone}', shell=True, check=True)

  # ---------------------------------------------------------------------------
  # Create a window per host & ssh directly into each one
  # ---------------------------------------------------------------------------

  if args.skip_ssh_connection:
    print("Skipping connecting to hosts - you indicated you already have separate tmux windows with each host connected.")
  else:
    for i in range(0, num_hosts-1):
      tmux("split-window -h")  # TODO

    # Layout the windows nicely
    tmux("select-layout tiled")

    # If you get stuck connecting here - make sure you can connect normally.
    for i in range(0, num_hosts):
      tmux_select_pane(i)
      tmux_shell(
          f"gcloud compute tpus tpu-vm ssh {args.tpu_name} --zone {args.zone} --project {args.project} --worker {i}")

    tmux_select_pane(0)
    # terminal broadcast across all panes
    tmux("setw synchronize-panes")

    # Run setup script
    tmux_shell("source ./setup.sh")

    print(f'{OKGREEN}Connected to all hosts {ENDC}')

  # ---------------------------------------------------------------------------
  # Set up the fswatch sync from local to all workers
  # ---------------------------------------------------------------------------

  if args.skip_setup_fswatch:
    print("Skipping fswatch - you're already syncing")
  else:
    tmux("new-window")
    tmux_shell(f"echo 'for ip in {' '.join(external_ips)}; do rsync -ravz --stats --exclude=.git {adhd_path} \$ip:~/adhd & done' > sync.sh")
    tmux_shell("chmod u+x sync.sh")
    tmux_shell(f"fswatch -0 -o {adhd_path} | xargs -0 -I {{}} ./sync.sh")
    tmux("select-window -t 0")

    print(f'{OKGREEN}Syncing! {ENDC}')

