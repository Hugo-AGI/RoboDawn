"""Query Vulkan physical-device properties without creating GPU resources."""

import ctypes as c
import hashlib
import json
import socket


class ApplicationInfo(c.Structure):
    _fields_ = [
        ("sType", c.c_uint32), ("pNext", c.c_void_p),
        ("pApplicationName", c.c_char_p), ("applicationVersion", c.c_uint32),
        ("pEngineName", c.c_char_p), ("engineVersion", c.c_uint32),
        ("apiVersion", c.c_uint32),
    ]


class InstanceCreateInfo(c.Structure):
    _fields_ = [
        ("sType", c.c_uint32), ("pNext", c.c_void_p), ("flags", c.c_uint32),
        ("pApplicationInfo", c.POINTER(ApplicationInfo)),
        ("enabledLayerCount", c.c_uint32), ("ppEnabledLayerNames", c.c_void_p),
        ("enabledExtensionCount", c.c_uint32), ("ppEnabledExtensionNames", c.c_void_p),
    ]


class PhysicalDeviceProperties(c.Structure):
    _fields_ = [
        ("apiVersion", c.c_uint32), ("driverVersion", c.c_uint32),
        ("vendorID", c.c_uint32), ("deviceID", c.c_uint32),
        ("deviceType", c.c_uint32), ("deviceName", c.c_char * 256),
        ("pipelineCacheUUID", c.c_uint8 * 16),
        # VkPhysicalDeviceLimits and SparseProperties are not decoded; reserve
        # aligned storage larger than their ABI sizes so they can be hashed.
        ("limitsAndSparsePropertiesStorage", c.c_uint64 * 256),
    ]


class Properties2(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                ("properties", PhysicalDeviceProperties)]


class Maintenance3(c.Structure):
    _fields_ = [
        ("sType", c.c_uint32), ("pNext", c.c_void_p),
        ("maxPerSetDescriptors", c.c_uint32), ("maxMemoryAllocationSize", c.c_uint64),
    ]


def probe() -> dict:
    vk = c.CDLL("libvulkan.so.1")
    vk.vkCreateInstance.argtypes = [c.POINTER(InstanceCreateInfo), c.c_void_p, c.POINTER(c.c_void_p)]
    vk.vkCreateInstance.restype = c.c_int32
    vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
    vk.vkEnumeratePhysicalDevices.restype = c.c_int32
    vk.vkGetPhysicalDeviceProperties2.argtypes = [c.c_void_p, c.POINTER(Properties2)]
    vk.vkGetPhysicalDeviceProperties2.restype = None
    vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    vk.vkDestroyInstance.restype = None

    app = ApplicationInfo(sType=0, pApplicationName=b"robodojo-vulkan-probe", apiVersion=(1 << 22) | (1 << 12))
    create = InstanceCreateInfo(sType=1, pApplicationInfo=c.pointer(app))
    instance = c.c_void_p()
    result = vk.vkCreateInstance(c.byref(create), None, c.byref(instance))
    if result:
        raise RuntimeError(f"vkCreateInstance returned {result}")
    try:
        count = c.c_uint32()
        result = vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None)
        if result:
            raise RuntimeError(f"vkEnumeratePhysicalDevices returned {result}")
        devices = (c.c_void_p * count.value)()
        result = vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices)
        if result:
            raise RuntimeError(f"vkEnumeratePhysicalDevices returned {result}")
        records = []
        other_devices = []
        for index, device in enumerate(devices):
            maintenance = Maintenance3(sType=1000168000)
            properties = Properties2(sType=1000059001, pNext=c.addressof(maintenance))
            vk.vkGetPhysicalDeviceProperties2(device, c.byref(properties))
            info = properties.properties
            if info.vendorID != 0x10DE:
                other_devices.append({
                    "device_index": index, "name": info.deviceName.decode(),
                    "vendor_id": info.vendorID, "device_type": info.deviceType,
                    "maxMemoryAllocationSize": maintenance.maxMemoryAllocationSize,
                })
                continue
            driver = info.driverVersion
            records.append({
                "device_index": index,
                "name": info.deviceName.decode(),
                "driver": [driver >> 22, (driver >> 14) & 255, (driver >> 6) & 255, driver & 63],
                "api_version": info.apiVersion,
                "device_id": info.deviceID,
                "properties_sha256": hashlib.sha256(bytes(info)).hexdigest(),
                "maxPerSetDescriptors": maintenance.maxPerSetDescriptors,
                "maxMemoryAllocationSize": maintenance.maxMemoryAllocationSize,
            })
        return {"host": socket.gethostname(), "devices": records, "other_devices": other_devices}
    finally:
        vk.vkDestroyInstance(instance, None)


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
