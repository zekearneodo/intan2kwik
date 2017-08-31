# intan2kwik
Convert files recorded with intan RHD2000 eval software (.rhd) to the Kwik format (.kwd, .kwe, etc)

For now, it just makes one .kwd file with all the .rhd files in a folder.
It creates one dataset (data) with all of the neural ports (in order) and subsequently the adc channels.
For the digital channels, it just creates another table (in the same group), named dig_in.

For both, it keeps track of the time in the tables t_data and t_dig.
