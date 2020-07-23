def GetDeviceTagAndMachineDeviceIds(parallel_conf):
    machine_device_ids = []
    for device_name in parallel_conf.device_name:
        machine_device_ids.append(device_name)
    device_tag = parallel_conf.device_tag
    return device_tag, machine_device_ids
