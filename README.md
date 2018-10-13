# intan2kwik
Convert files recorded with intan RHD2000/RHS2000 eval software (.rhd/.rhs) to the Klusta-team's Kwik format (.kwd, .kwe, etc) (https://github.com/klusta-team/kwiklib/wiki/Kwik-format)

For now, it just makes one .kwd file with all the .rhd files in a folder.
It creates one dataset (data) with all of the neural ports (in order) and subsequently the adc channels.
For the digital channels, it just creates another table (in the same group), named dig_in.

For both, it keeps track of the time in the tables t_data and t_dig.
The extraction of other data/metadata is in progress.

Examples:
- Read, sort and append all .rhd files in a folder into a .kwd file, stripping all the amplifier and the board_adc channels into the rec 0 group:

``` python
first_header = intan_to_kwd(rhd_folder, kwd_file_path, rec=0, include_channels=['amplifier', 'board_adc'], 
                        board='rhd')
```

- Same, but with data recorded in the .rhs format:
``` python
first_header = intan_to_kwd(rhs_folder, kwd_file_path, rec=0, include_channels=['amplifier', 'board_adc'], 
                        board='rhs')
```

The returned first_header is the header of the first file in the folder, as a dictionary containing the structure described in the intan specification for the files .rhd and .rhs (http://intantech.com/files/Intan_RHD2000_data_file_formats.pdf and http://intantech.com/files/Intan_RHS2000_data_file_formats.pdf, respectively).
