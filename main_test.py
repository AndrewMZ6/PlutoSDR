import devices

# if mode='single' then sdrrx=sdrtx, otherwise sdrrx = 'NOBEARD'
sdrtx, sdrrx = devices.initialize_sdr(single_mode=[5], tx='FISHER', swap=False)


# get user bits and time domain data
transmitted_user_bits, tx_data = generate_tx_data(params)


sdrtx.tx(tx_data)
#    v
#    |
#    |
#    |
#    |
#    |
#    v
receivced_data = sdrrx.rx()


# demodualted user data
received_user_bits = process_data(receivced_data, show_graphs=False)


# compare transmitted bits and demodulated bits
show_bit_errors(received_user_bits, transmitted_user_bits)