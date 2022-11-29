import devices
import data_gen
import data_processing

# if mode='single' then sdrrx=sdrtx, otherwise sdrrx = 'NOBEARD'
sdrtx, sdrrx = devices.initialize_sdr(single_mode=True, tx='FISHER', swap=False)


# get user bits and time domain data, or generate data for DPD calcualtion
transmitted_user_bits, tx_data = data_gen.generate_tx_data(frames=5)


sdrtx.tx(tx_data)
#    v
#    |
#    |
#    |
#    |
#    |
#    v
receivced_data = sdrrx.rx(0)


# demodualted user data1
received_user_bits = data_processing.process_data(receivced_data, show_graphs=False)

exit()
# compare transmitted bits and demodulated bits
data_processing.show_bit_errors(received_user_bits, transmitted_user_bits)