
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Final
# local
from data_loader.data_loader import DataLoader
from data_loader.data_loader import Columns
from data_loader.ecg_loader import EcgLoader
from data_loader.ecg_loader import EcgColumns
from data_loader.file_settings import SELECT_HEALTH
from change_detection.detector import calculate_signal_discrepancy


# constants
COLUMN_WEEKLY_AVG: Final[str] = 'weekly_avg'
COLUMN_DISCREPANCY: Final[str] = 'discrepancy'
COLUMN_ZERO: Final[str] = 'zeros'


def show_burned_energy(filepath: str) -> None:
    # Start, End 日付選択
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input('ここから', datetime(2020, 1, 1))

    with date_col2:
        end_date = st.date_input('ここまで', datetime(2024, 3, 15))

    # ファイル読み込み
    loader = DataLoader(filepath)
    loader.load()
    df = loader.get(column='', time_from=start_date, time_to=end_date)  # type: ignore
    week_df = df.copy().set_index(Columns.Date).resample('W').mean(numeric_only=True).reset_index()
    merged_df = pd.concat([df, week_df.rename(columns={Columns.EnergyBurned: COLUMN_WEEKLY_AVG})]).reset_index()

    # plot burned energy
    st.write("##")
    st.subheader('消費カロリー (毎日, 1週間平均)')

    fig = px.line(
        merged_df, x=Columns.Date, y=[Columns.EnergyBurned, COLUMN_WEEKLY_AVG],
        # title='Burned Energy',
        color_discrete_map={
            Columns.EnergyBurned: 'skyblue',
            COLUMN_WEEKLY_AVG: 'orange'})
    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Burned Energy'),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', orientation='v'))
    st.plotly_chart(fig)

    # calculate discrepancy
    st.write("##")
    st.subheader('Discrepancy')

    window_size = st.slider('window size', 10, 100, 50, 10)
    gamma_str = st.select_slider(
        'parameter gamma',
        value='1.000',
        options=[f'{(10**(i-3)):.3f}' for i in range(5)])
    gamma = float(gamma_str)  # type: ignore

    energy_array = df[Columns.EnergyBurned].to_numpy()
    energy_array /= energy_array.max()
    discrepancy = calculate_signal_discrepancy(
        energy_array, 'rbf', {'window_size': window_size, 'gamma': gamma})
    df[COLUMN_DISCREPANCY] = discrepancy
    df[COLUMN_ZERO] = [0.] * len(df)

    # plot discrepancy
    # # energy
    fig2 = px.line(
        merged_df, x=Columns.Date, y=[Columns.EnergyBurned, COLUMN_WEEKLY_AVG],
        # title='Burned Energy',
        color_discrete_map={
            Columns.EnergyBurned: 'skyblue',
            COLUMN_WEEKLY_AVG: 'orange'})
    # # RBF discrepancy
    fig2.add_trace(go.Scatter(x=df[Columns.Date], y=df[COLUMN_DISCREPANCY], name='value', mode='none', fill='tonexty', fillcolor='rgba(255, 0, 0, 0.6)', yaxis='y2'))
    fig2.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Burned Energy'),
        yaxis2=dict(title='Discrepancy', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', orientation='v'))
    st.plotly_chart(fig2)


def show_ecg(filepath: str) -> None:
    # ファイル読み込み
    loader = EcgLoader(filepath)
    loader.load()
    df = loader.get()

    # frame rate
    st.write(f'Sampling Rate {loader.get_sampling_rate():.2f} [Hz]')

    # show plot
    fig = px.line(df, x=EcgColumns.Time, y=EcgColumns.Value)
    fig.update_layout(
        xaxis=dict(title='Time [sec]'),
        yaxis=dict(title='V [uV]'),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', orientation='v'))
    st.plotly_chart(fig)


def main():
    # make Streamlit
    st.title('Health Check')

    # 見たいものを選択
    target_health_key = st.selectbox(label='どれを見たい ?', options=SELECT_HEALTH.keys())
    filepath = SELECT_HEALTH[target_health_key]

    match target_health_key:
        case 'Energy':
            show_burned_energy(filepath)
        case 'ECG':
            show_ecg(filepath)
        case _:
            # do nothing
            pass


if __name__ == '__main__':
    main()
