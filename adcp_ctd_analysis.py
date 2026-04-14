import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, hilbert
import pywt
import gsw
import os

# =========================================================
# 1. ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# =========================================================
def julian_to_datetime(jd):
    return datetime.fromordinal(int(jd)) + timedelta(days=jd % 1) - timedelta(days=366)

# =========================================================
# 2. ПУТИ К ФАЙЛАМ
# =========================================================
base_path = r"C:\Документы\ДИПЛОМ\Химченко_данные\adcp_ctd"

adcp_files = [
    os.path.join(base_path, "230610_0000-20_2359.txt"),
    os.path.join(base_path, "230601_0000-10_2359.txt"),
    os.path.join(base_path, "230621_0000-30_2359.txt")
]

ctd_file = os.path.join(base_path, "CTD.txt")

# =========================================================
# 3. ЧТЕНИЕ ADCP
# =========================================================
adcp_list = []

for file in adcp_files:
    df = pd.read_csv(file, sep='\t', encoding='cp1251')

    for col in ['Ve', 'Vn', 'Vz', 'Depth']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Ve', 'Vn'])
    df['datetime'] = df['iTimeDbl'].apply(julian_to_datetime)

    adcp_list.append(df)

adcp = pd.concat(adcp_list).sort_values('datetime').reset_index(drop=True)

# =========================================================
# 4. ЧТЕНИЕ CTD
# =========================================================
ctd = pd.read_csv(ctd_file, sep=',', encoding='cp1251')

ctd['datetime'] = pd.to_datetime(ctd['DATE'] + ' ' + ctd['TIME'], errors='coerce')

ctd = ctd.dropna(subset=[
    'Temp',
    'Sal(psu)',
    'Pressure(psia)',
    'Depth(m)',
    'GPSLatitude',
    'GPSLongitude',
    'datetime'
])

# =========================================================
# 5. ЧАСТОТА ВЯЙСЯЛЯ–БРЕНТА (CTD)
# =========================================================
ctd = ctd.sort_values('Depth(m)').reset_index(drop=True)
ctd = ctd.loc[ctd['Depth(m)'].diff().fillna(1) != 0]

# psia → dbar
p = ctd['Depth(m)'].values  # грубое, но устойчивое приближение

SA = gsw.SA_from_SP(
    ctd['Sal(psu)'].values,
    p,
    ctd['GPSLongitude'].values,
    ctd['GPSLatitude'].values
)

CT = gsw.CT_from_t(SA, ctd['Temp'].values, p)
rho = gsw.rho(SA, CT, p)

drho_dz = np.gradient(rho, ctd['Depth(m)'].values)

g = 9.81
rho0 = 1025

N2 = -g / rho0 * drho_dz
N = np.sqrt(np.clip(N2, 0, None))

plt.figure(figsize=(5,7))
plt.plot(N, ctd['Depth(m)'], linewidth=1.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.xlabel('Частота Вяйсяля–Брента N, рад/с')
plt.ylabel('Глубина, м')
plt.title('Вертикальный профиль частоты Вяйсяля–Брента по данным CTD')
plt.show()
plt.close('all')

# =========================================================
# 6. ФИЛЬТРАЦИЯ ADCP (5–15 МИН)
# =========================================================
fs = 1/60  # Гц

low = 1 / (15*60)
high = 1 / (5*60)

b, a = butter(2, [low/(fs/2), high/(fs/2)], btype='band')

adcp['Ve_f'] = filtfilt(b, a, adcp['Ve'])
adcp['Vn_f'] = filtfilt(b, a, adcp['Vn'])

plt.figure(figsize=(10,4))
plt.plot(adcp['datetime'], adcp['Ve_f'], label='Зональная компонента (Ve)', linewidth=0.7)
plt.plot(adcp['datetime'], adcp['Vn_f'], label='Меридиональная компонента (Vn)', linewidth=0.7)
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Время')
plt.ylabel('Скорость течения, м/с')
plt.title('Короткопериодная изменчивость компонент скорости течений (5–15 мин)')
plt.show()
plt.close('all')

# =========================================================
# 7. СПЕКТР МОЩНОСТИ (ИСХОДНЫЙ СИГНАЛ)
# =========================================================
Ve = adcp['Ve'].values
Ve = Ve - np.mean(Ve)

f, Pxx = welch(
    Ve,
    fs=fs,
    window='hann',
    nperseg=min(4096, len(Ve)//2),
    detrend='linear'
)

mask = f > 0
T = 1 / f[mask] / 60

plt.figure(figsize=(8,5))
plt.loglog(T, Pxx[mask], linewidth=1.5)
plt.gca().invert_xaxis()
plt.grid(True, which='both')
plt.xlabel('Период, мин')
plt.ylabel('Спектральная плотность мощности')
plt.title('Спектр мощности зональной компоненты скорости течений (Welch)')
plt.show()
plt.close('all')

# =========================================================
# 8. ВЕЙВЛЕТ-АНАЛИЗ (ЦУГИ ВОЛН)
# =========================================================
step = 4

Ve_w = adcp['Ve_f'].values[::step]
dt = 60 * step

periods = np.linspace(5, 15, 40)
scales = periods * 60 / (2 * np.pi)

coeffs, _ = pywt.cwt(Ve_w, scales, 'morl', sampling_period=dt)

t0 = np.datetime64(adcp['datetime'].iloc[0])
t = (adcp['datetime'].iloc[::step].to_numpy(dtype='datetime64[s]') - t0) / np.timedelta64(1, 's')

plt.figure(figsize=(12, 4))
plt.pcolormesh(t, periods, np.abs(coeffs), shading='auto', cmap='jet')
plt.gca().invert_yaxis()
plt.colorbar(label='Амплитуда')
plt.xlabel('Время, с')
plt.ylabel('Период, мин')
plt.title('Вейвлет-анализ (5–15 мин)')
plt.show()
plt.close('all')

# =========================================================
# 9. ОЦЕНКА НАПРАВЛЕНИЯ РАСПРОСТРАНЕНИЯ ВОЛН
# =========================================================
Ve_h = hilbert(adcp['Ve_f'])
Vn_h = hilbert(adcp['Vn_f'])

phase_diff = np.angle(Vn_h) - np.angle(Ve_h)

direction = np.degrees(np.arctan2(
    np.sin(phase_diff),
    np.cos(phase_diff)
))

plt.figure(figsize=(10,3))
plt.plot(adcp['datetime'], direction, linewidth=0.8)
plt.grid(True)
plt.xlabel('Время')
plt.ylabel('Фазовый угол, град')
plt.title('Оценка направления распространения короткопериодных волн')
plt.show()
plt.close('all')
