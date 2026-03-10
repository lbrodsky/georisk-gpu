# GeoRisk GPU access
## 1. Desktop PC
* Start PC with automatically run into Windows 11
* Start linux requires to select boot source (F11) and than NVME#0: Samsung SSD 990 PRO 2TB

## 2. SSH Access 
SSH allows secure remote login to the workstation.
```
ssh username@SERVER_IP 

e.g. 
ssh lukas@georisk-gpu 
```
After connecting you will be placed in your home directory.
ToBeDone when CIT assign fixed IP. 

Note: 
SSH key authentication! 
- Password login should be avoided when possible.
- Generate key (local machine) 
```
ssh-keygen -t name 
```
Default location: ~/.ssh/id_name
- Copy key to server (Linux) 
``` 
ssh-copy-id lukas@georisk-gpu
```
- Login without password 
```
ssh lukas@georisk-gpu
```

SSH config: 
- To simplify login, add an entry to your local SSH config:

~/.ssh/config

```
Host georisk-gpu
    HostName SERVER_IP
    User username
    IdentityFile ~/.ssh/id_name
```
- Then connect simply with:
```
ssh georisk-gpu
```

## 3. Remote desktop 
Do we need? 
