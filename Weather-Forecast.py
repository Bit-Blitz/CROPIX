import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_coordinates(city, state):
    try:
        geolocator = Nominatim(user_agent="weather_app_v1")
        location_string = f"{city}, {state}"
        print(f"Searching for coordinates for: {location_string}...")
        location = geolocator.geocode(location_string)
        if location:
            print(f"Found at ({location.latitude:.4f}, {location.longitude:.4f})")
            return location.latitude, location.longitude
        else:
            print("Location could not be found.")
            return None, None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Error: Geocoding service failed. {e}")
        return None, None

def get_weather_data(latitude, longitude, start_date, end_date):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS10M",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    print("Fetching weather data from NASA POWER API...")
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        print("Successfully retrieved data.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from NASA POWER API: {e}")
        return None

def display_weather_data(data, city, state):
    if not data or 'properties' not in data or 'parameter' not in data['properties']:
        print("Weather data is not in the expected format or is empty.")
        return

    print("\n" + "="*50)
    print(f"30-Day Weather Overview for {city.title()}, {state.title()}")
    print("="*50)

    params = data['properties']['parameter']
    dates = sorted(params['T2M'].keys())
    fill_value = data.get('header', {}).get('fill_value', -999)

    for date_str in dates:
        display_day(date_str, params, fill_value)

    print("="*50)
    print("\nParameter Key:")
    print("  - Temp: Avg Temperature at 2m (°C)")
    print("  - Max/Min: Max/Min Temperature at 2m (°C)")
    print("  - RH: Relative Humidity at 2m (%)")
    print("  - Precip: Precipitation (mm/day)")
    print("  - Wind: Wind Speed at 10m (m/s)")
    print("\nNote: NASA POWER provides satellite and model data, not a traditional forecast.")
    print("Data has a latency of a few days, so future dates may show as 'not available'.\n")
    
def display_day(date_str, params, fill_value):
    try:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        formatted_date = date_obj.strftime('%Y-%m-%d (%A)')

        temp = params['T2M'].get(date_str, fill_value)
        temp_max = params['T2M_MAX'].get(date_str, fill_value)
        temp_min = params['T2M_MIN'].get(date_str, fill_value)
        rh = params['RH2M'].get(date_str, fill_value)
        precip = params['PRECTOTCORR'].get(date_str, fill_value)
        ws = params['WS10M'].get(date_str, fill_value)

        if temp == fill_value:
            print(f"{formatted_date}: Data not available")
            return
            
        print(f"{formatted_date}:")
        print(f"  - Temp: {temp:.2f}°C  (Max: {temp_max:.2f}°C, Min: {temp_min:.2f}°C)")
        print(f"  - RH: {rh:.2f}%")
        print(f"  - Precip: {precip:.2f} mm/day")
        print(f"  - Wind: {ws:.2f} m/s\n")
    except (ValueError, KeyError) as e:
        print(f"Could not process data for date {date_str}: {e}")

def plot_weather_data(data, city, state):
    if not data or 'properties' not in data or 'parameter' not in data['properties']:
        print("Cannot plot data, as it is not in the expected format or is empty.")
        return

    params = data['properties']['parameter']
    dates_str = sorted(params['T2M'].keys())
    fill_value = data.get('header', {}).get('fill_value', -999)

    plot_dates, temps, temps_max, temps_min = [], [], [], []
    for date_str in dates_str:
        if params['T2M'].get(date_str, fill_value) != fill_value:
            plot_dates.append(datetime.strptime(date_str, '%Y%m%d'))
            temps.append(params['T2M'][date_str])
            temps_max.append(params['T2M_MAX'][date_str])
            temps_min.append(params['T2M_MIN'][date_str])

    if not plot_dates:
        print("No valid temperature data available to plot.")
        return

    today = datetime.now()
    last_past_index = -1
    for i, date in enumerate(plot_dates):
        if date < today:
            last_past_index = i
        else:
            break

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    if last_past_index != -1:
        past_dates = plot_dates[:last_past_index+2]
        past_max = temps_max[:last_past_index+2]
        past_min = temps_min[:last_past_index+2]
        ax.plot(past_dates, past_max, 'r-', label='Past Max Temp')
        ax.plot(past_dates, past_min, 'b-', label='Past Min Temp')
        ax.fill_between(past_dates, past_min, past_max, color='gray', alpha=0.2, label='Past Temp Range')

    if last_past_index < len(plot_dates) - 1:
        future_dates = plot_dates[last_past_index+1:]
        future_max = temps_max[last_past_index+1:]
        future_min = temps_min[last_past_index+1:]
        ax.plot(future_dates, future_max, 'r--', label='Projected Max Temp')
        ax.plot(future_dates, future_min, 'b--', label='Projected Min Temp')
        ax.fill_between(future_dates, future_min, future_max, color='gray', alpha=0.1, label='Projected Temp Range')

    ax.set_title(f"Temperature Overview for {city.title()}, {state.title()}", fontsize=16)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend()
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = "temperature_plot.png"
    plt.savefig(filename)
    print(f"\nTemperature plot saved as '{filename}'")


if __name__ == "__main__":
    city = input("Please enter the city name: ")
    state = input("Please enter the state name: ")

    latitude, longitude = get_coordinates(city, state)

    if latitude is not None and longitude is not None:
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        end_date = (datetime.now() + timedelta(days=30)).strftime('%Y%m%d')

        weather_data = get_weather_data(latitude, longitude, start_date, end_date)
        if weather_data:
            display_weather_data(weather_data, city, state)
            plot_weather_data(weather_data, city, state)

