
def geometric_adstock(data, decay_rate, max_duration):
    adstock_values = []
    for i in range(len(data)):
        adstock_value = data[i]
        for j in range(1, min(i+1, max_duration+1)):
            adstock_value += data[i-j]*decay_rate**j
        adstock_values.append(adstock_value)
    return adstock_values