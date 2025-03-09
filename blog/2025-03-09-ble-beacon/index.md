---
slug: ble-beacon
title: 使用Python检测蓝牙信号
authors: eason
tags: [bluetooth]
draft: false
---

本文介绍了如何使用Python在MacOS上检测BLE信号并可视化展示信号强度。

<!-- truncate -->

## 常见的定位方法

在设备定位的领域内，有大概3种定位技术，其分别为：UWB（超宽带）、BLE（低功耗蓝牙）和WiFi。这三种技术各有优缺点，适用于不同的场景。

### UWB（Ultra-Wideband，超宽带）定位

UWB是一种使用极短脉冲在宽频带上传输数据的无线通信技术。在定位领域，UWB具有以下特点：

#### 工作原理
UWB定位主要基于TOF（Time of Flight，飞行时间）或TDOA（Time Difference of Arrival，到达时间差）原理。设备通过测量无线电信号从发射到接收的时间来计算距离，进而确定位置。

#### 优势
- **高精度**：UWB可以提供厘米级的定位精度（通常在10-30厘米范围内）
- **抗多径干扰**：由于使用极短的脉冲，UWB对多径干扰有很强的抵抗力
- **穿墙能力强**：UWB信号可以穿透墙壁和其他障碍物
- **低功耗**：相对于其他高精度定位技术，UWB的功耗较低

#### 劣势
- **成本高**：UWB设备和基础设施的成本相对较高
- **覆盖范围有限**：通常需要多个锚点（基站）来覆盖较大区域
- **标准化程度较低**：虽然有IEEE 802.15.4z标准，但市场上的实现多样化

#### 应用场景
- 高精度室内定位
- 资产追踪
- 智能家居
- 工业自动化
- 车辆防盗系统（如Apple AirTag等）

### BLE（Bluetooth Low Energy，低功耗蓝牙）定位

BLE是蓝牙技术的一个子集，专为低功耗应用设计。在定位领域，BLE主要通过信标（Beacon）技术实现。

#### 工作原理
BLE定位主要基于RSSI（Received Signal Strength Indication，接收信号强度指示）。通过测量接收到的信号强度，并结合路径损耗模型，可以估算设备与信标之间的距离。常见的协议包括iBeacon（苹果）和Eddystone（谷歌）。

#### 优势
- **低功耗**：BLE设备可以使用纽扣电池运行数月甚至数年
- **成本低**：BLE芯片和信标价格便宜，部署成本低
- **兼容性好**：几乎所有现代智能手机都支持BLE
- **部署简单**：无需复杂的基础设施

#### 劣势
- **精度有限**：典型精度在3-5米，受环境影响大
- **易受干扰**：信号容易受到人体、墙壁等障碍物的影响
- **距离有限**：有效范围通常在50米以内

#### 应用场景
- 商场导航
- 展览会信息推送
- 资产追踪
- 考勤系统
- 智能家居自动化

### WiFi定位

WiFi定位利用现有的WiFi基础设施进行室内定位，是最广泛部署的室内定位技术之一。

#### 工作原理
WiFi定位主要有两种方式：
1. **基于RSSI的三边测量**：通过测量设备与多个WiFi接入点之间的信号强度，估算距离并确定位置
2. **指纹定位**：预先采集空间中各点的WiFi信号特征，形成"指纹数据库"，定位时将实时采集的信号与数据库匹配

#### 优势
- **基础设施广泛**：利用现有WiFi网络，无需额外硬件
- **覆盖范围大**：单个接入点可覆盖数十米范围
- **成本低**：如果已有WiFi网络，几乎无额外成本
- **兼容性好**：几乎所有移动设备都支持WiFi

#### 劣势
- **精度一般**：典型精度在3-15米，取决于环境和接入点密度
- **易受干扰**：信号受环境变化影响大
- **功耗较高**：相比BLE和UWB，WiFi的功耗较高
- **初始化复杂**：指纹定位需要前期大量采集工作

#### 应用场景
- 大型建筑物内导航
- 商场客流分析
- 公共场所位置服务
- 资产管理
- 智能办公

### 三种技术对比

| 技术 | 精度 | 功耗 | 成本 | 覆盖范围 | 抗干扰能力 |
|------|------|------|------|----------|------------|
| UWB  | 10-30厘米 | 中等 | 高 | 小（~50米） | 强 |
| BLE  | 3-5米 | 低 | 低 | 中（~50米） | 弱 |
| WiFi | 3-15米 | 高 | 低（利用现有网络） | 大（~100米） | 中 |

在实际应用中，这三种技术往往会结合使用，以弥补各自的不足。例如，可以使用WiFi进行粗略定位，然后使用BLE进行区域确认，最后在需要高精度的场景下使用UWB进行精确定位。

接下来，我们将重点介绍如何使用Python检测BLE信号并可视化展示信号强度。

## Python检测BLE信号

在本节中，我们将介绍如何使用Python来检测和分析BLE信号。我们将基于[yishi-projects/ble-beacon](https://github.com/yishi-projects/ble-beacon)项目中的代码来实现这一功能。

### 所需库和依赖

首先，我们需要安装以下Python库：

```bash
pip install bleak kafka-python
```

主要依赖包括：
- **bleak**：跨平台的BLE客户端库，支持Windows、macOS和Linux
- **kafka-python**：用于将数据发送到Kafka（可选，用于数据流处理）

### 代码结构

我们的BLE检测程序主要包含以下几个部分：

1. 初始化和配置
2. BLE设备扫描
3. 信标数据解析（iBeacon、Eddystone等）
4. 数据处理和可视化

### 初始化和配置

首先，我们需要导入必要的库并设置基本配置：

```python
import asyncio
from bleak import BleakScanner
import uuid
import time
import datetime
import os
import configparser

# 全局变量控制扫描状态
_scanning_active = False

# 加载配置文件
def load_config():
    """从~/.ble/config.conf加载配置，如果不存在则创建默认配置"""
    config = configparser.ConfigParser()
    
    # 默认配置
    config['kafka'] = {
        'broker': 'localhost:9092',
        'topic': 'ble_beacons'
    }
    
    # 创建配置目录（如果不存在）
    config_dir = os.path.expanduser("~/.ble")
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, "config.conf")
    
    # 如果配置文件存在，读取它
    if os.path.exists(config_file):
        config.read(config_file)
    else:
        # 创建默认配置文件
        with open(config_file, 'w') as f:
            config.write(f)
    
    return config
```

### BLE设备扫描

BLE设备扫描是整个程序的核心部分。我们使用`bleak`库的`BleakScanner`来异步扫描周围的BLE设备：

```python
async def scan_ble_devices():
    """扫描BLE设备并处理信标数据"""
    # 获取主机ID（用于数据标识）
    host_id = get_host_id()
    
    # 计数器（用于日志）
    scan_count = 0
    
    # 设置扫描状态
    global _scanning_active
    _scanning_active = True
    
    try:
        # 持续扫描循环
        while _scanning_active:
            scan_count += 1
            
            # 扫描设备（超时1秒）
            devices = await BleakScanner.discover(timeout=1.0)
            
            # 检查是否应该停止扫描
            if not _scanning_active:
                break
            
            # 处理每个设备
            beacons_found = 0
            for device in devices:
                # 提取制造商数据
                if device.metadata.get('manufacturer_data'):
                    for company_code, data in device.metadata['manufacturer_data'].items():
                        # 检测不同类型的信标
                        process_beacon_data(company_code, data, device)
            
            # 等待下一次扫描
            await asyncio.sleep(0)
    finally:
        # 清理资源
        pass
```

### 信标数据解析

BLE信标有多种类型，最常见的是iBeacon（苹果）和Eddystone（谷歌）。我们需要根据不同的协议格式解析数据：

```python
def process_beacon_data(company_code, data, device):
    """根据不同的信标类型解析数据"""
    # 检查iBeacon（苹果公司代码是0x004C）
    if company_code == 0x004C and len(data) >= 23:
        try:
            # 检查iBeacon标识符（0x02, 0x15）
            if data[0] == 0x02 and data[1] == 0x15:
                # 解析iBeacon数据
                uuid_bytes = data[2:18]
                uuid_str = str(uuid.UUID(bytes=bytes(uuid_bytes)))
                major = int.from_bytes(data[18:20], byteorder='big')
                minor = int.from_bytes(data[20:22], byteorder='big')
                tx_power = data[22] - 256 if data[22] > 127 else data[22]
                
                beacon_data = {
                    'uuid': uuid_str,
                    'major': major,
                    'minor': minor,
                    'tx_power': tx_power,
                    'rssi': device.rssi,
                    'address': device.address,
                    'name': device.name or 'Unknown'
                }
                
                # 处理iBeacon数据
                process_beacon('iBeacon', beacon_data)
        except Exception as e:
            print(f"处理iBeacon数据时出错: {e}")
    
    # 检查Eddystone信标（谷歌公司代码是0x00AA）
    elif company_code == 0x00AA and len(data) >= 20:
        try:
            # 检查Eddystone标识符
            if data[0] == 0xAA and data[1] == 0xFE:
                frame_type = data[2]
                
                # Eddystone-UID
                if frame_type == 0x00:
                    namespace = bytes(data[3:13]).hex()
                    instance = bytes(data[13:19]).hex()
                    
                    beacon_data = {
                        'namespace': namespace,
                        'instance': instance,
                        'rssi': device.rssi,
                        'address': device.address,
                        'name': device.name or 'Unknown'
                    }
                    
                    # 处理Eddystone-UID数据
                    process_beacon('Eddystone-UID', beacon_data)
                
                # Eddystone-URL
                elif frame_type == 0x10:
                    url_scheme = ['http://www.', 'https://www.', 'http://', 'https://'][data[3]]
                    url_data = bytes(data[4:]).decode('ascii')
                    url = url_scheme + url_data
                    
                    beacon_data = {
                        'url': url,
                        'rssi': device.rssi,
                        'address': device.address,
                        'name': device.name or 'Unknown'
                    }
                    
                    # 处理Eddystone-URL数据
                    process_beacon('Eddystone-URL', beacon_data)
        except Exception as e:
            print(f"处理Eddystone数据时出错: {e}")
```

### 数据处理和可视化

收集到的BLE信号数据可以通过多种方式进行处理和可视化：

1. **实时显示**：使用GUI库（如Tkinter、PyQt等）实时显示检测到的设备和信号强度
2. **数据存储**：将数据保存到本地文件或数据库中
3. **数据流处理**：使用Kafka等消息队列进行实时数据流处理
4. **信号强度可视化**：使用matplotlib等库绘制信号强度热图或时间序列图

以下是一个简单的数据处理函数示例：

```python
def process_beacon(beacon_type, beacon_data):
    """处理信标数据"""
    timestamp = datetime.datetime.now().isoformat()
    
    # 添加通用字段
    message = {
        'type': beacon_type,
        'timestamp': timestamp,
        'rssi': beacon_data.get('rssi', 0),
        'address': beacon_data.get('address', 'unknown')
    }
    
    # 添加特定类型的字段
    message.update(beacon_data)
    
    # 这里可以添加数据处理逻辑
    # 例如：保存到文件、发送到服务器、更新GUI等
    
    return message
```

### 完整示例

下面是一个简单但完整的BLE扫描器示例，它会扫描周围的BLE设备并打印出检测到的信标信息：

```python
import asyncio
from bleak import BleakScanner
import uuid
import datetime

async def main():
    print("开始扫描BLE设备...")
    
    # 扫描设备
    devices = await BleakScanner.discover(timeout=5.0)
    
    print(f"发现 {len(devices)} 个设备")
    
    # 处理每个设备
    for device in devices:
        print(f"设备: {device.address} ({device.name or 'Unknown'}), RSSI: {device.rssi}")
        
        # 提取制造商数据
        if device.metadata.get('manufacturer_data'):
            for company_code, data in device.metadata['manufacturer_data'].items():
                # 检查iBeacon
                if company_code == 0x004C and len(data) >= 23:
                    if data[0] == 0x02 and data[1] == 0x15:
                        # 解析iBeacon数据
                        uuid_bytes = data[2:18]
                        uuid_str = str(uuid.UUID(bytes=bytes(uuid_bytes)))
                        major = int.from_bytes(data[18:20], byteorder='big')
                        minor = int.from_bytes(data[20:22], byteorder='big')
                        
                        print(f"  iBeacon: UUID={uuid_str}, Major={major}, Minor={minor}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 信号强度可视化

为了更直观地展示BLE信号强度，我们可以使用matplotlib库创建可视化图表。以下是一个简单的示例，展示如何绘制信号强度随时间变化的图表：

```python
import matplotlib.pyplot as plt
import numpy as np
import time
import asyncio
from bleak import BleakScanner

async def monitor_device(address, duration=60):
    """监控特定设备的信号强度"""
    timestamps = []
    rssi_values = []
    
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        # 扫描设备
        devices = await BleakScanner.discover(timeout=1.0)
        
        # 查找目标设备
        for device in devices:
            if device.address == address:
                # 记录时间和RSSI
                timestamps.append(time.time() - start_time)
                rssi_values.append(device.rssi)
                print(f"时间: {timestamps[-1]:.1f}s, RSSI: {rssi_values[-1]} dBm")
                break
        
        # 等待下一次扫描
        await asyncio.sleep(0.5)
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, rssi_values, 'b-')
    plt.xlabel('时间 (秒)')
    plt.ylabel('信号强度 (dBm)')
    plt.title(f'设备 {address} 的BLE信号强度')
    plt.grid(True)
    plt.savefig('ble_signal_strength.png')
    plt.show()

# 使用示例
# asyncio.run(monitor_device('XX:XX:XX:XX:XX:XX', 60))
```

## 总结

通过Python和bleak库，我们可以轻松地检测和分析BLE信号。这种方法适用于多种应用场景，如室内定位、资产追踪、存在检测等。

在实际应用中，我们可以根据需要扩展上述代码，例如：

- 添加距离估算（基于RSSI和路径损耗模型）
- 实现三边测量定位算法
- 开发实时监控仪表板
- 集成机器学习算法进行模式识别

BLE信标技术结合Python的灵活性，为我们提供了一个强大的工具，可以用于构建各种智能空间应用。

## 参考资料

- [yishi-projects/ble-beacon](https://github.com/yishi-projects/ble-beacon) - BLE信标检测项目
- [Bleak文档](https://bleak.readthedocs.io/) - 跨平台BLE客户端库
- [蓝牙SIG](https://www.bluetooth.com/) - 蓝牙技术标准

