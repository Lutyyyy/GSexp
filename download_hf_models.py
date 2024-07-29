from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    revision="1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9",
    filename="v1-5-pruned.ckpt",
    local_dir="models",
)

hf_hub_download(
    repo_id="lllyasviel/ControlNet-v1-1",
    revision="69fc48b9cbd98661f6d0288dc59b59a5ccb32a6b",
    filename="control_v11f1e_sd15_tile.pth",
    local_dir="models",
)

# download dataset
# https://drive.google.com/file/d/1-1Ko4SMah3JimyPVsSVioCfqZBLxuu9Q/view?usp=drive_link
# https://drive.google.com/file/d/1-03m4nSOLTQciKkhglaKKvRCYavM0neM/view?usp=drive_link
# curl -H "Authorization: Bearer ya29.a0AXooCgsfZEn3QeV9gZY2ARVPWhqcHsSWeLmO75KZon6NRMK0uvFYD0iEMG5qgU1-UrH27dlfXaX5R7xRMkfhAAxb9MjSQcUJH_CCsZUlRDdPvR2EqtNbJgihFnyDo78oWA3-pQa5g4_wGSZSTMm9d0ehPqKBAA_OEW2jaCgYKAVsSARISFQHGX2Misj_bEddFD3lEqWOUsY_y9g0171" https://www.googleapis.com/drive/v3/files/1-1Ko4SMah3JimyPVsSVioCfqZBLxuu9Q?alt=media -o mip360.zip
# curl -H "Authorization: Bearer ya29.a0AXooCgsfZEn3QeV9gZY2ARVPWhqcHsSWeLmO75KZon6NRMK0uvFYD0iEMG5qgU1-UrH27dlfXaX5R7xRMkfhAAxb9MjSQcUJH_CCsZUlRDdPvR2EqtNbJgihFnyDo78oWA3-pQa5g4_wGSZSTMm9d0ehPqKBAA_OEW2jaCgYKAVsSARISFQHGX2Misj_bEddFD3lEqWOUsY_y9g0171" https://www.googleapis.com/drive/v3/files/1-03m4nSOLTQciKkhglaKKvRCYavM0neM?alt=media -o omi360.zip
# ssh-keygen -t rsa               # 运行命令后一直回车即可生成 SSH-Key
# cp /root/.ssh/id_rsa.pub ./  # 将公钥拷贝到当期目录
# rsync -avuzP -e "ssh -p 21334 -o StrictHostKeyChecking=no" *.zip root@connect.bjb1.seetacloud.com:/root/autodl-tmp
