Create and activate venv

## run the following Commands in order:
pip install -r requirements.txt
python -u "d:\CROPIX\SYNTHETIC DATA Scripts\IMAGE_PROCESSOR[CNN].py"
python -u "d:\CROPIX\SYNTHETIC DATA Scripts\Labels_initializer.py"
python -u "d:\CROPIX\SYNTHETIC DATA Scripts\synthetic_data[fertilizer_recom].py"
python -u "d:\CROPIX\SYNTHETIC DATA Scripts\SyntheticData[MARKET ANALYSIS].py"
python -u "d:\CROPIX\Models\Crop-Yield[XGB].py"
python -u "d:\CROPIX\Models\Fertilizer-Recomm[Random Forest].py"
python -u "d:\CROPIX\Models\Market-Forecast[SARIMA].py"
python -u "d:\CROPIX\Models\Soil-Crop-Recomm[KNN].py"
python -u "d:\CROPIX\Models\Weather-Forecast[LSTM].py"