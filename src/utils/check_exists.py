    start_time = pd.to_datetime(trange[0]).strftime('%Y%m%d_%H%M%S')
    end_time = pd.to_datetime(trange[1]).strftime('%Y%m%d_%H%M%S')

    fname = os.path.join(data_dir, f'{start_time}_{end_time}.h5')

    # Check whether file already exists or not
    if not os.path.exists(fname):
        flag = 1
    else:
        flag = 0
        print("File already exists. Skipping to avoid overwrite...")