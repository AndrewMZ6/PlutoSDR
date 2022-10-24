Help on function demodulate in module commpy.modulation:

ddeemmoodduullaattee(self, input_symbols, demod_type, noise_var=0)
    Demodulate (map) a set of constellation symbols to corresponding bits.
    
    Parameters
    ----------
    input_symbols : 1D ndarray of complex floats
        Input symbols to be demodulated.
    
    demod_type : string
        'hard' for hard decision output (bits)
        'soft' for soft decision output (LLRs)
    
    noise_var : float
        AWGN variance. Needs to be specified only if demod_type is 'soft'
    
    Returns
    -------
    demod_bits : 1D ndarray of ints
        Corresponding demodulated bits.
