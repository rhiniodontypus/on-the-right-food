"""
This script establishes a connection to a Virtual Machine (VM)
on the Google Cloud Platform (GCP) and allows you to download 
any file from the specified directory.
"""

import os
import sys
import shutil
import paramiko
from google.cloud import storage

""" 
Type here the external IP address of your virtual machine in the format of 0.0.0.0
You can also leave it empty.
"""
VM_IP = ""
# type your username of your virtual machine
VM_USERNAME = "user"
VM_HOME_PATH = "/home/user/"
# specify your ssh folder
LOCAL_PATH_PRIVATE_KEY = "/home/user/.ssh/google_compute_engine"
 # specify your local folder to download to
LOCAL_DOWNLOAD_PATH = "./"     
""" 
Type here the name of the folders you want to access on your virtual machine. 
You can define as many folders as you like.
"""
VM_DIR = ['otrf-training',
          'otrf-training/data/',
          'otrf-training/output/'
]


def ip_settings():
    VM_IP = input("External IP of your VM: ")
    return VM_IP


def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.RSAKey.from_private_key_file(LOCAL_PATH_PRIVATE_KEY)
    client.connect(hostname=VM_IP, username=VM_USERNAME, pkey=key)
    return client


def select_directories():
    print("")
    print("Select your directory: ")
    print("")
    j = 0
    for dir in VM_DIR:
        j += 1
        print(f"[{j}] {dir}")
        
    print("")
    dir_key = input("Please choose the number of your directory: ")
    dir_path = VM_HOME_PATH + VM_DIR[int(dir_key) - 1]
    print
    print("Selected directory: ", dir_path)
    return dir_path


def get_file_list():
    stdin, stdout, stderr = client.exec_command(f'ls {dir_path}')
    output = stdout.readlines()
    file_list_vm = []
    for files_vm in output:
        file_list_vm.append(files_vm.replace('\n',''))
        

    all_files_vm = []
    for i in file_list_vm:
        if "." in i or "last_checkpoint" in i:
            all_files_vm.append(i.split("."))
            
    all_files_vm = sorted(all_files_vm, key = lambda x: x[-1])

    print("")
    print(f"The folder {dir_path} has the entries: \n")

    file_list_sorted = []
    k = 0
    for files in all_files_vm:
        k += 1
        print(f"[{k}] {'.'.join(files)}")
        file_list_sorted.append('.'.join(files))
    return file_list_sorted


def select_files():
    print("")
    select_file_keys = input("Which file(s) do you want to copy?\nYou can choose multiple files seperated by comma (,), e.g. 1,2,3: ")
    print("")
    print("You selected the following file(s): \n")
    all_files_vm_final = []
    for l in select_file_keys.split(","):
        print(file_list_sorted[int(l)-1])
        all_files_vm_final.append(file_list_sorted[int(l)-1])
    print("")
    cont = input("Do you want to continue? [y/N] ")
    return all_files_vm_final, cont

def copy_files():
    ftp_client = client.open_sftp()
    for f_vm in all_files_vm_final:
        print(f"Copy {dir_path}" + f"{f_vm} from VM to local {LOCAL_DOWNLOAD_PATH}")
        ftp_client.get(dir_path + f_vm, LOCAL_DOWNLOAD_PATH + f_vm)
    ftp_client.close()


if __name__ == "__main__":
    os.system('clear')
    if len(VM_IP) == 0:
        VM_IP = ip_settings()
    client = ssh_connect()
    dir_path = select_directories()
    file_list_sorted = get_file_list()
    all_files_vm_final, cont = select_files()

    while cont != "y":
        print("You have canceled the file transfer.")
        all_files_vm_final, cont = select_files()
        if cont == "y":
            print("Selected files are copied to your local folder!")
            print("")
            copy_files()
            break
    else:
        print("Selected files are copied to your local folder!")
        print("")
        copy_files()

    print("SSH client is closed!")
    client.close()