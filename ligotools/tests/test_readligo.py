from ligotools import readligo as rl


def test_loaddata_H1():
	strain, time, dq = rl.loaddata('ligotools/tests/ligo_data/H-H1_LOSC_4_V2-1126259446-32.hdf5', 'H1')
	assert (len(strain) == 131072) & (len(time) == 131072) & (len(dq) == 13)
	
def test_loaddata_L1():
	strain, time, dq = rl.loaddata('ligotools/tests/ligo_data/H-H1_LOSC_4_V2-1126259446-32.hdf5', 'L1')
	assert (len(strain) == 131072) & (len(time) == 131072) & (len(dq) == 13)
	
def test_dq_channel_to_seglist_CBC_CAT3():
	strain, time, chan_dict = rl.loaddata('ligotools/tests/ligo_data/L-L1_LOSC_4_V2-1126259446-32.hdf5', 'H1')
	DQflag = 'CBC_CAT3'
	segment_list = rl.dq_channel_to_seglist(chan_dict[DQflag])
	assert segment_list[0] == slice(0, 131072, None)

def test_dq_channel_to_seglist_NO_CBC_HW_INJ():
	strain, time, chan_dict = rl.loaddata('ligotools/tests/ligo_data/L-L1_LOSC_4_V2-1126259446-32.hdf5', 'H1')
	segment_list = rl.dq_channel_to_seglist(chan_dict['NO_CBC_HW_INJ'])
	assert segment_list[0] == slice(0, 131072, None)