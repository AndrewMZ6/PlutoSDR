
# if mode='single' then sdrrx=sdrtx, otherwise sdrrx = 'NOBEARD'
sdrtx, sdrrx = initialize_sdr(mode='single', tx='BEARD')

# preambula contains double sequence
preambula = generate_preambula()

# user_data contains user data (duh...)
user_data = generate_data(frames_num = 10)

# concatenate preambula and user data to transmit 
tx_data = preambula + user_data

sdrtx.tx(tx_data)
#    v
#    |
#    |
#    |
#    |
#    |
#    v
receivced_data = sdrrx.rx()

# corr object contains information about cut indexes
corr = correlate(preambula, receivced_data)

# cut out preambula and user data from received data using information containing in corr object
preambula_part1, preambula_part2, user_data_rx = cutout_data(receivced_data, corr)

# find frequency shift with part1 and part2 of preambula and use it on user_data_rx
user_data_rx = freq_shift_recovery(preambula_part1, preambula_part2, user_data_rx)


# extract frames from user_data_rx, and equalize each frame based on it's pilots
equlized_data: tuple = equalize_frame(user_data_rx, frames=10)

# 