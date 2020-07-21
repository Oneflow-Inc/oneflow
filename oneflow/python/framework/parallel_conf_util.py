def GetDeviceTagAndMachineDeviceIds(parallel_conf):
    machine_device_ids = []
    for device_name in parallel_conf.device_name:
        machine_id, device_tag, device_ids = device_name.split(":")
        machine_device_ids.append("{}:{}".format(machine_id, device_ids))
    return device_tag, machine_device_ids
