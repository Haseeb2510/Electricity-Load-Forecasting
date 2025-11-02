import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

if __name__ == '__main__':
    print("\nWelcome to the main menu of this project! \n\nSelect option you want to chose:")
    options = """
1. Evaluation
2. EDA
3. Train XGBoost model
4. Train LSTM model
5. Train TFT (Temporal Frusion Transformer) model
"""

    option = input(options + '\nOption: ')

    if option == '1':
        from evaluation import main as eva_main
        eva_main()

    elif option == '2':
        from EDA import main as eda_main
        eda_main()
        
    elif option == '3':
        from XGBoost_model_training import main as xgb_main
        xgb_main()

    elif option == '4':
        from LSTM_model_training import main as lstm_main
        lstm_main()

    elif option == '5':
        from TFT_model_training import main as tft_main
        tft_main()
    
    else:
        print('Option NOT VALID.')
    