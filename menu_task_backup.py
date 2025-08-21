import os
import secrets
import subprocess
import urllib.request

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import pywhatkit as pwk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from twilio.rest import Client

# Optional libs (may not be installed in your env)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import boto3
except Exception:
    boto3 = None

# ===== API Key Setup =====
API_KEY = secrets.token_hex(8)
print(f"\n🔑 Your session API key is: {API_KEY}")
print("Use this API key in the input field to access app features.\n")

# ===== Twilio Config =====
TWILIO_SID = "YOUR_TWILIO_SID"
TWILIO_AUTH = "YOUR_TWILIO_AUTH"
TWILIO_PHONE = "+160868941"
client = Client(TWILIO_SID, TWILIO_AUTH)

# ===== Haarcascade Download =====
HAAR_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(HAAR_PATH):
    url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, HAAR_PATH)

# ===== Command Maps (Linux, Docker, Kubernetes) =====
command_map = {
    # --- Linux ---
    "🖥 Linux: Show Date": "date",
    "🖥 Linux: Show Calendar": "cal",
    "🖥 Linux: List Files": "ls -lh",
    "🖥 Linux: Current Directory": "pwd",
    "🖥 Linux: Disk Usage": "df -h",
    "🖥 Linux: Check Memory Usage": "free -m",
    "🖥 Linux: CPU Info": "lscpu",
    "🖥 Linux: Memory Info": "cat /proc/meminfo",
    "🖥 Linux: Uptime": "uptime",
    "🖥 Linux: List Users": "cut -d: -f1 /etc/passwd",
    "🖥 Linux: Show IP Address": "ip a",
    "🖥 Linux: Show Hostname": "hostname",
    "🖥 Linux: Running Processes": "ps aux",
    "🖥 Linux: Environment Variables": "printenv",
    "🖥 Linux: System Architecture": "uname -m",
    "🖥 Linux: Kernel Version": "uname -r",
    "🖥 Linux: Operating System Info": "uname -a",
    "🖥 Linux: List Logged In Users": "who",
    "🖥 Linux: Show Last 5 Logins": "command -v last >/dev/null && last -n 5 || echo '❌ last not installed'",
    "🖥 Linux: List Open Ports": "ss -tuln",
    "🖥 Linux: Top Running Processes": "top -bn1 | head -20",
    "🖥 Linux: Find OS Type": "cat /etc/os-release",
    "🖥 Linux: Network Interfaces": "ip link show",
    "🖥 Linux: Ping Google": "ping -c 4 google.com",
    "🖥 Linux: Mounted Filesystems": "mount | column -t",
    "🖥 Linux: Show Crontab": "crontab -l",
    "🖥 Linux: List Hidden Files": "ls -la",
    "🖥 Linux: Active Network Connections": "netstat -tulnp || ss -tulnp",
    "🖥 Linux: List USB Devices": "lsusb",
    "🖥 Linux: List PCI Devices": "lspci",
    "🖥 Linux: Show Disk Partitions": "lsblk",
    "🖥 Linux: Disk Space (ncdu)": "command -v ncdu >/dev/null && ncdu / || echo '❌ ncdu not installed'",
    "🖥 Linux: Check File Descriptors": "ulimit -n",
    "🖥 Linux: Check Open Files": "lsof | head -20",
    "🖥 Linux: Show System Logs": "journalctl -n 20",
    "🖥 Linux: Check SSH Status": "systemctl status ssh || service ssh status",
    "🖥 Linux: Current User": "whoami",
    "🖥 Linux: File Type of bash": "file /bin/bash",
    "🖥 Linux: Check SELinux Status": "sestatus || echo 'sestatus not found'",
    "🖥 Linux: View dmesg logs": "dmesg | tail -20",
    "🖥 Linux: Count Logged Users": "who | wc -l",
    "🖥 Linux: List Active Services": "systemctl list-units --type=service --state=running",
    "🖥 Linux: View Aliases": "alias",
    "🖥 Linux: Find a File": "find /etc -name 'hosts' 2>/dev/null",
    "🖥 Linux: Show PATH": "echo $PATH",
    "🖥 Linux: Reboot System": "echo 'Run: sudo reboot (permission required)'",
    "🖥 Linux: Shutdown System": "echo 'Run: sudo shutdown now (permission required)'",
    "🖥 Linux: Update Packages (Debian)": "sudo apt update && echo '✅ Updated (Debian)' || echo '❌ apt not found'",
    "🖥 Linux: Update Packages (RHEL)": "sudo yum update -y && echo '✅ Updated (RHEL)' || echo '❌ yum not found'",
    "🖥 Linux: Docker Version": "docker version || echo '❌ Docker not installed'",
    "🖥 Linux: Docker Containers": "docker ps -a || echo '❌ Docker not installed'",
    # --- Docker ---
    "🐳 Docker: List Running Containers": "docker ps",
    "🐳 Docker: List All Containers": "docker ps -a",
    "🐳 Docker: List Images": "docker images",
    "🐳 Docker: System Info": "docker info",
    "🐳 Docker: Disk Usage": "docker system df",
    "🐳 Docker: Container Stats": "docker stats --no-stream",
    "🐳 Docker: Docker Version": "docker version",
    # --- Kubernetes ---
    "☸️ Kubernetes: Get Pods": "kubectl get pods",
    "☸️ Kubernetes: Get Nodes": "kubectl get nodes",
    "☸️ Kubernetes: Get Services": "kubectl get svc",
    "☸️ Kubernetes: Get Deployments": "kubectl get deployments",
    "☸️ Kubernetes: Describe Pod": "kubectl describe pod <pod_name>",
    "☸️ Kubernetes: View Pod Logs": "kubectl logs <pod_name>",
    "☸️ Kubernetes: Apply Config File": "kubectl apply -f <file.yaml>",
    "☸️ Kubernetes: Delete Pod": "kubectl delete pod <pod_name>",
    "☸️ Kubernetes: Scale Deployment": "kubectl scale deployment <deployment_name> --replicas=<num>",
    "☸️ Kubernetes: View Kube Config": "kubectl config view",
    "☸️ Kubernetes: Get StatefulSets": "kubectl get statefulsets",
    "☸️ Kubernetes: Get ReplicaSets": "kubectl get rs",
    "☸️ Kubernetes: Describe Node": "kubectl describe node <node_name>"
}

# ===== API Key Decorator =====
def require_key(func):
    def wrapper(*args):
        entered_key = args[-1]
        if entered_key != API_KEY:
            return "❌ Invalid API Key"
        return func(*args[:-1])
    return wrapper

# ===== Combined SSH Handler with Port and Error Output =====
@require_key
def run_ssh_combined(command_choice, username, ip, port):
    command = command_map.get(command_choice)
    if not command:
        return "❌ Invalid command selected."
    # If user provided a port placeholder, include it in ssh if not default 22
    ssh_base = f"ssh -p {port} {username}@{ip}" if port and str(port) != "22" else f"ssh {username}@{ip}"
    ssh_cmd = f"{ssh_base} {command}"
    try:
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return f"❌ SSH Failed:\n{result.stderr.strip() or 'No error message'}\n\n⚠ Please check:\n- SSH service running\n- IP/username/port correct\n- SSH key setup (no password prompts)"
        output = result.stdout.strip()
        return output if output else "✅ Command executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "⏰ SSH connection timed out. Is the server up and reachable?"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

# ===== Twilio Tasks =====
@require_key
def send_sms(phone):
    try:
        msg = client.messages.create(to=phone, from_=TWILIO_PHONE, body="Hello from your app!")
        return f"✅ SMS sent to {phone}: {msg.sid}"
    except Exception as e:
        return f"❌ SMS Error: {str(e)}"

@require_key
def make_call(phone):
    try:
        call = client.calls.create(to=phone, from_=TWILIO_PHONE, url="http://demo.twilio.com/docs/voice.xml")
        return f"📞 Call initiated to {phone}: {call.sid}"
    except Exception as e:
        return f"❌ Call Error: {str(e)}"

# ===== WhatsApp =====
@require_key
def send_whatsapp(phone, message):
    try:
        pwk.sendwhatmsg_instantly(phone, message, wait_time=10, tab_close=True)
        return "✅ WhatsApp message sent using pywhatkit"
    except Exception as e:
        return f"❌ WhatsApp error: {str(e)}"

# ===== Face Swap =====
@require_key
def face_swap(img1, img2):
    try:
        face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
        if len(faces1) == 0 or len(faces2) == 0:
            return "❌ Face not detected in one or both images."
        x1, y1, w1, h1 = faces1[0]
        x2, y2, w2, h2 = faces2[0]
        face1 = img1[y1:y1+h1, x1:x1+w1]
        face1_resized = cv2.resize(face1, (w2, h2))
        swapped = img2.copy()
        swapped[y2:y2+h2, x2:x2+w2] = face1_resized
        path = "face_swapped.jpg"
        cv2.imwrite(path, swapped)
        return path
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ===== Sketch Image =====
@require_key
def sketch_image(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        path = "sketch.png"
        cv2.imwrite(path, sketch)
        return path
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ===== Dockerfile AI Guide =====
def dockerfile_ai_guide(question, gemini_key):
    if genai is None:
        return "❌ google.generativeai (genai) library not installed."
    if not gemini_key:
        return "❌ Please enter your Gemini API Key."
    if "dockerfile" not in question.lower():
        return "⚠ This AI guide is restricted to Dockerfile-related queries only."
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Answer this only about Dockerfile best practices and usage: {question}")
        # response.text or response.candidates[0].content may vary by genai version; guard it:
        return getattr(response, "text", str(response))
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# ===== Machine Learning Utility (returns file path or None, plus message) =====
@require_key
def machine_learning_tools(file, action):
    try:
        if file is None:
            return None, "❌ No file uploaded."

        df = pd.read_csv(file.name)

        if action == "Visualize Data":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return None, "❌ No numeric columns found to visualize."
            plt.figure(figsize=(8, 5))
            if len(numeric_cols) >= 2:
                sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
            else:
                df[numeric_cols].hist(figsize=(8, 5))
            plt.tight_layout()
            path = "ml_visual.png"
            plt.savefig(path)
            plt.close()
            return path, "✅ Visualization created successfully."

        elif action == "Data Preprocessing":
            # Fill missing values
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")
                else:
                    df[col] = df[col].fillna(df[col].mean())

            # Encode categorical variables
            label_enc = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = label_enc.fit_transform(df[col].astype(str))
                except Exception:
                    pass

            # Normalize numeric columns
            scaler = MinMaxScaler()
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                df[num_cols] = scaler.fit_transform(df[num_cols])

            # Save processed CSV
            processed_path = "processed_data.csv"
            df.to_csv(processed_path, index=False)
            return processed_path, "✅ Data preprocessing completed successfully."

        else:
            return None, "❌ Invalid action."

    except Exception as e:
        return None, f"❌ ML Tab Error: {str(e)}"

# ===== AWS Actions =====
@require_key
def aws_actions(aws_access_key, aws_secret_key, region, action, resource_name, instance_type, api_key_placeholder):
    """
    Note: last parameter is the app's API_KEY (checked by decorator).
    This function accepts:
    - aws_access_key, aws_secret_key, region: credentials
    - action: "EC2 - Launch Instance" or "S3 - Create Bucket" etc.
    - resource_name: bucket name / s3 name / lambda name / etc.
    - instance_type: used for EC2 instance type
    """
    if boto3 is None:
        return "❌ boto3 not installed or available."

    if not aws_access_key or not aws_secret_key:
        return "❌ Please provide AWS credentials."

    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region or "us-east-1"
        )

        if action == "EC2 - Launch Instance":
            ec2 = session.resource("ec2")
            # NOTE: AMI id is region dependent. This is an example for us-east-1 (change per region).
            # You should replace this with a valid AMI for your region or make AMI selectable.
            example_ami = "ami-0c02fb55956c7d316"  # Amazon Linux 2 (may need update)
            itype = instance_type or "t2.micro"
            inst = ec2.create_instances(ImageId=example_ami, MinCount=1, MaxCount=1, InstanceType=itype)
            return f"✅ EC2 Instance launched: {inst[0].id}"

        elif action == "S3 - Create Bucket":
            if not resource_name:
                return "❌ Please provide a bucket name in 'Resource Name'."
            s3 = session.client("s3")
            # bucket creation parameters vary by region
            create_kwargs = {"Bucket": resource_name}
            if session.region_name != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": session.region_name}
            s3.create_bucket(**create_kwargs)
            return f"✅ S3 Bucket '{resource_name}' created successfully."

        elif action == "Lambda - Deploy Function":
            return "⚠ Lambda deployment not implemented (placeholder)."

        elif action == "RDS - Create Database Instance":
            return "⚠ RDS creation not implemented (placeholder)."

        else:
            return "❌ Unknown AWS action."

    except Exception as e:
        return f"❌ AWS Error: {str(e)}"

# ===== Gradio Interfaces =====

# SSH UI
ssh_ui = gr.Interface(
    fn=run_ssh_combined,
    inputs=[
        gr.Dropdown(choices=list(command_map.keys()), label="📋 Choose a Task"),
        gr.Text(label="👤 Remote Username", placeholder="e.g., ubuntu"),
        gr.Text(label="🌐 Remote Server IP", placeholder="e.g., 192.168.1.10"),
        gr.Text(label="🔌 SSH Port", value="22"),
        gr.Text(label="🔑 Enter App API Key", type="password")
    ],
    outputs="text",
    title="🔧 Remote System Assistant",
    description="Run Linux / Docker / Kubernetes commands remotely. Make sure SSH works without asking for password."
)

# SMS UI
sms_ui = gr.Interface(
    fn=send_sms,
    inputs=[
        gr.Text(label="📱 Phone Number (+CountryCode)"),
        gr.Text(label="🔑 App API Key", type="password")
    ],
    outputs="text",
    title="📨 Send SMS"
)

# Call UI
call_ui = gr.Interface(
    fn=make_call,
    inputs=[
        gr.Text(label="📞 Phone Number (+CountryCode)"),
        gr.Text(label="🔑 App API Key", type="password")
    ],
    outputs="text",
    title="📞 Make a Call"
)

# WhatsApp UI
wa_ui = gr.Interface(
    fn=send_whatsapp,
    inputs=[
        gr.Text(label="📱 WhatsApp Number with Country Code"),
        gr.Text(label="💬 Message Text"),
        gr.Text(label="🔑 App API Key", type="password")
    ],
    outputs="text",
    title="💬 Send WhatsApp Message"
)

# Face swap UI
face_ui = gr.Interface(
    fn=face_swap,
    inputs=[
        gr.Image(label="🖼 Upload Image 1 (with Face)"),
        gr.Image(label="🖼 Upload Image 2 (to Replace Face)"),
        gr.Text(label="🔑 App API Key", type="password")
    ],
    outputs="image",
    title="🔀 Face Swap"
)

# Sketch UI
sketch_ui = gr.Interface(
    fn=sketch_image,
    inputs=[
        gr.Image(label="🖼 Upload Image"),
        gr.Text(label="🔑 App API Key", type="password")
    ],
    outputs="image",
    title="🎨 Convert Image to Sketch"
)

# Dockerfile AI UI
docker_ai_ui = gr.Interface(
    fn=dockerfile_ai_guide,
    inputs=[
        gr.Textbox(label="❓ Ask about Dockerfile"),
        gr.Text(label="🔑 Gemini API Key (Google)", type="password")
    ],
    outputs="text",
    title="🤖 Dockerfile AI Guide"
)

# Machine Learning UI (file + status message)
ml_ui = gr.Interface(
    fn=machine_learning_tools,
    inputs=[
        gr.File(label="📂 Upload CSV File", file_types=[".csv"]),
        gr.Radio(choices=["Visualize Data", "Data Preprocessing"], label="Select Action"),
        gr.Text(label="🔑 App API Key", type="password")
    ],
    outputs=[
        gr.File(label="📄 Processed CSV / Image Output"),
        gr.Textbox(label="ℹ️ Status / Message")
    ],
    title="📊 Machine Learning Tools"
)

# AWS UI
aws_ui = gr.Interface(
    fn=aws_actions,
    inputs=[
        gr.Text(label="🔐 AWS Access Key ID"),
        gr.Text(label="🔑 AWS Secret Access Key", type="password"),
        gr.Text(label="🌍 AWS Region", value="us-east-1"),
        gr.Dropdown(choices=["EC2 - Launch Instance", "S3 - Create Bucket", "Lambda - Deploy Function", "RDS - Create Database Instance"], label="Select AWS Action"),
        gr.Text(label="🏷 Resource Name (Bucket name / Lambda name / etc.)", placeholder="e.g., my-bucket-name"),
        gr.Text(label="🖥 EC2 Instance Type (if launching EC2)", value="t2.micro"),
        gr.Text(label="🔑 App API Key", type="password")
    ],
    outputs="text",
    title="☁️ AWS Service Launcher",
    description="Use boto3 to create basic AWS resources. Make sure credentials are correct and have required permissions."
)

# ===== Final Launch =====
gr.TabbedInterface(
    [ssh_ui, sms_ui, call_ui, wa_ui, face_ui, sketch_ui, docker_ai_ui, ml_ui, aws_ui],
    ["🧠 Remote Tasks", "📨 SMS", "📞 Call", "💬 WhatsApp", "🔀 Face Swap", "🎨 Sketch", "🤖 Dockerfile AI Guide", "📊 ML Tools", "☁️ AWS"]
).launch()
