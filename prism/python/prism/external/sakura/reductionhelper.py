import contextlib
import itertools
import multiprocessing
import os
import Queue
from threading import *
import time
import re
import numpy

#import libsakurapy
#import _casasakura
libsakurapy = None

#from taskinit import gentools, ms#, ssd
#import mtpy
#import reductionhelper_util as rhutil

#libsakurapy.initialize()

def dbgPrint(msg):
	#print(msg)
	pass

class Context(object):
	# Attributes are inQ, inCv, outQ, outCv, qLen, pendingItems
	pass

class EndOfDataException(BaseException):
	pass

EOD = EndOfDataException() # singleton instance. Use 'is' to compare.

def worker(func, context):
	try:
		while True:
			item = None
			with context.inCv:
				while True:
					try:
						item = context.inQ.get(False)
						if item is EOD:
							raise item
						break
					except Queue.Empty:
						context.inCv.wait()
			try:
				result = func(item)
			except Exception as e:
				result = e
			with context.outCv:
				context.outQ.put(result)
				context.outCv.notify()
	except EndOfDataException:
		pass
	finally:
		thr_id = current_thread().ident
		dbgPrint("{0} terminated".format(thr_id))

# out of order and parallel execution generator
def paraMap(numThreads, func, dataSource):
	assert numThreads > 0
	context = Context()
	context.qLen = int(numThreads * 1.5)
	assert context.qLen >= numThreads
	context.inQ = Queue.Queue(maxsize=context.qLen)
	context.inCv = Condition()
	context.outQ = Queue.Queue(maxsize=context.qLen)
	context.outCv = Condition()
	context.pendingItems = 0
	threads = []
	for i in range(numThreads):
		thr = Thread(target=worker, args=(func, context))
		thr.daemon = True
		thr.start()
		threads.append(thr)
	def fillInQ(context):
		try:
			while context.pendingItems < context.qLen:
				item = dataSource.next()
				with context.inCv:
					context.inQ.put(item, False)
					context.pendingItems += 1
					context.inCv.notify()
		except Queue.Full:
			assert False
	def putEODIntoInQ(context):
		try:
			with context.inCv:
				context.inQ.put(EOD, False)
				context.pendingItems += 1
				context.inCv.notify()
		except Full:
			assert False
	def getFromOutQ(context):
		assert 0 < context.pendingItems and context.pendingItems <= context.qLen
		with context.outCv:
			while True:
				try:
					item = context.outQ.get(False)
					context.pendingItems -= 1
					return item
				except Queue.Empty:
					context.outCv.wait()
	try:
		fillInQ(context)
		assert 0 < context.pendingItems and context.pendingItems <= context.qLen
		while context.pendingItems > 0:
			item = getFromOutQ(context)
			assert 0 <= context.pendingItems and context.pendingItems < context.qLen
			try:
				fillInQ(context)
				assert 0 < context.pendingItems and context.pendingItems <= context.qLen
			finally:
				yield item
	except StopIteration as e:
		while context.pendingItems > 0:
			yield getFromOutQ(context)
	assert context.pendingItems == 0
	for i in range(numThreads):
		assert context.pendingItems < context.qLen
		putEODIntoInQ(context)

# class GenerateQueryHelper(object):
#     def __init__(self, vis, data_desc_id, antenna_id, \
# 		 field, spw, timerange, antenna, scan, observation, msselect):
#         self.vis = vis
#         if field is None: field = ''
#         if spw is None: spw = ''
#         if timerange is None: timerange = ''
#         if antenna is None: antenna = ''
#         if scan is None: scan = ''
#         if observation is None: observation = ''
#         if msselect is None: msselect = ''
# 
#         self.selected_idx = {}
# 
#         try:
#             baseline_arg = '%s&&&'%(antenna) if str(antenna).strip()!='' else ''
#             self.selected_idx = ms.msseltoindex(vis=vis, field=field, \
#                                                 spw=spw, time=timerange, \
#                                                 baseline=baseline_arg, \
#                                                 scan=scan, \
#                                                 observation=observation, \
#                                                 taql=msselect)
#             self.valid_selection = self.is_effective(data_desc_id, antenna_id)
#         except:
#             self.valid_selection = False
# 
#     def is_effective(self, data_desc_id, antenna_id):
#         return self.is_effective_id('spwdd', data_desc_id) and \
# 	    self.is_effective_id('antenna1', antenna_id) and \
# 	    self.is_effective_id('antenna2', antenna_id)
# 
#     def is_effective_id(self, key, value):
#         res = True
#         if len(self.selected_idx[key]) > 0:
#             try:
#                 list(self.selected_idx[key]).index(value)
#             except:
#                 res = False
#         return res
# 
#     def get_taql(self, state_id, data_desc_id, antenna_id, timerange, msselect):
#         elem = []
#         self._append_taql(elem, 'STATE_ID', 'IN', state_id)
#         self._append_taql(elem, 'DATA_DESC_ID', '==', data_desc_id)
#         self._append_taql(elem, 'ANTENNA1', '==', antenna_id)
#         self._append_taql(elem, 'ANTENNA2', '==', antenna_id)
#         self._append_taql(elem, 'FIELD_ID', 'IN', 'field', True)
#         if timerange != '':
#             taql_timerange = rhutil.select_by_timerange(self.vis, timerange)
#             self._append_taql(elem, '', '', taql_timerange, True)
#         self._append_taql(elem, 'SCAN_NUMBER', 'IN', 'scan', True)
#         self._append_taql(elem, 'OBSERVATION_ID', 'IN', 'obsids', True)
#         self._append_taql(elem, '', '', msselect, True)
#         return ' && '.join(elem)
# 
#     def _append_taql(self, elem, keyword, operand, value, check=False):
#         ope = operand.strip()
# 	if isinstance(value, str): value = value.strip()
#         can_append = True
#         if check:
#             if ope == 'IN':
#                 can_append = len(self.selected_idx[value]) > 0
#             else:
#                 can_append = (value is not None) and (str(value) != '')
#         if can_append:
#             mgn = ' ' if ope != '' else ''
#             if check:
#                 val = str(self.selected_idx[value]) if ope == 'IN' else str(value)
#             else:
#                 val = str(value)
#             res = '(' + keyword.strip().upper() + mgn + ope + mgn + val + ')'
# 	    elem.append(res)
# 
#     def get_channel_selection(self):
#         idx_channel = self.selected_idx['channel']
#         return str(idx_channel) if len(idx_channel) > 0 else ''
# 
#     def get_pol_selection(self, pol):
#         return pol if pol is not None else ''
# 
# def generate_query(vis, field=None, spw=None, timerange=None, antenna=None, scan=None, pol=None, observation=None, msselect=None):
#     res_list = []
# 
#     with opentable(os.path.join(vis, 'DATA_DESCRIPTION')) as tb:
#         num_data_desc_id = tb.nrows()
# 
#     with opentable(os.path.join(vis, 'ANTENNA')) as tb:
#         num_antenna_id = tb.nrows()
# 
#     with opentable(os.path.join(vis, 'STATE')) as tb:
#         state_list = tb.getcol('OBS_MODE')
#         ondata_state_id = []
#         for i in xrange(len(state_list)):
#             if state_list[i].startswith('OBSERVE_TARGET#ON_SOURCE'):
#                 ondata_state_id.append(i)
# 
#     for data_desc_id, antenna_id in itertools.product(xrange(num_data_desc_id), xrange(num_antenna_id)):
#         gqh = GenerateQueryHelper(vis, data_desc_id, antenna_id, field, spw, timerange, antenna, scan, observation, msselect)
#         if gqh.valid_selection:
#             res = gqh.get_taql(ondata_state_id, data_desc_id, antenna_id, timerange, msselect)
#             #yield res, gqh.get_channel_selection(), gqh.get_pol_selection(pol)
#             res_list.append((res, gqh.get_channel_selection(), gqh.get_pol_selection(pol)))
#     return res_list
# 
# def get_context(query, spwidmap, ctx):
#     ddid, antennaid, chan_selection, pol_selection = _parse_query(query)
#     spwid = spwidmap[ddid]
#     c = ctx[spwid]
#     return c[0][antennaid], c[1], c[2], c[3], chan_selection, pol_selection
# 
# def _parse_query(query):
#     ddid = None
#     antennaid = None
#     elem = query[0].split(' && ')
#     for i in xrange(len(elem)):
#         elem_list = elem[i].replace('(', '').replace(')', '').split(' == ')
#         if elem_list[0].upper() == 'DATA_DESC_ID':
#             ddid = int(elem_list[1])
#         elif elem_list[0].upper() == 'ANTENNA1':
#             antennaid = int(elem_list[1])
#     assert (ddid is not None) and (antennaid is not None)
#     return ddid, antennaid, query[1], query[2]

# @contextlib.contextmanager
# def opentable(vis):
#     tb = gentools(['tb'])[0]
#     tb.open(vis, nomodify=False)
#     yield tb
#     tb.close()
# 
# @contextlib.contextmanager
# def openms(vis):
#     ms = gentools(['ms'])[0]
#     ms.open(vis)
#     yield ms
#     ms.close()
    
# def optimize_thread_parameters(table, query, spwmap):
#     try:
#         num_threads = min(3, multiprocessing.cpu_count())
#         assert num_threads > 0
# 
# 	subt = table.query(query[0])
#         num_rows = subt.nrows()
#         valid_ddid = str(_parse_query(query)[0])
#         if (num_rows > 0) and spwmap.has_key(valid_ddid):
#             data = subt.getcell('FLOAT_DATA', 0)
#             num_pols = len(data)
#             num_channels = len(data[0])
#             data_size_per_record = num_pols * num_channels * (8 + 1) * 2 * 10 #dummy
#             assert data_size_per_record > 0
# 
#             mem_size = 32*1024*1024*1024 #to be replaced with an appropriate function
#             num_record = mem_size / num_threads / data_size_per_record
#             if (num_record > num_rows): num_record = num_rows
#         else:
#             num_record = 0
# 
#         ###
#         #if num_record > 0: num_record = 300
#         ###
#         return num_record, num_threads
#     finally:
#         subt.close()
# 
# def readchunk(table, criteria, nrecord, ctx):
#     tb = table.query(criteria)
#     nrow = tb.nrows()
#     #print 'readchunk : nrow='+str(nrow)
#     rownumbers = tb.rownumbers()
#     tb.close()
#     nchunk = nrow / nrecord 
#     for ichunk in xrange(nchunk):
#         start = ichunk * nrecord
#         end = start + nrecord
#         chunk = _readchunk(table, rownumbers[start:end], ctx)
#         #print 'readchunk:',chunk
#         yield chunk
# 
#     # residuals
#     residual = nrow % nrecord
#     if residual > 0:
#         start = nrow - residual
#         end = nrow
#         chunk = _readchunk(table, rownumbers[start:end], ctx)
#         #print 'readchunk:',chunk
#         yield chunk
#         
# def _readchunk(table, rows, ctx):
#     #print '_readchunk: reading rows %s...'%(rows)
#     return tuple((_readrow(table, irow, ctx) for irow in rows))
# 
# def _readrow(table, row, ctx):
#     get = lambda col: table.getcell(col, row)
#     return (row, get('FLOAT_DATA'), get('FLAG'), get('TIME_CENTROID'), ctx)

# def reducechunk(chunk):
#     #print 'reducechunk'
#     return tuple((reducerecord(record) for record in chunk))
# 
# def reducerecord(record):
#     in_row, in_data, in_mask, in_time, in_context = record
#     #print 'reducing row %s'%(in_row)
# 
#     npol = len(in_data)
#     nchan = len(in_data[0])
#     out_data = numpy.ndarray([npol, nchan], dtype=numpy.float)
#     out_mask = numpy.ndarray([npol, nchan], dtype=numpy.bool)
# 
#     datatime = _casasakura.tosakura_double(numpy.array([in_time]))[0][0]
#     ctxcal, ctxbl, ctxsm, ctxmc, chan_selection, pol_selection = in_context
# 
#     #####<temporary start>--------
#     pol_list = xrange(npol)
#     #####<temporary end>--------
#     #####<should-be start>--------
#     #pol_list = parse_idx_selection(pol_selection, npol)
#     #####<should-be end>--------
# 
#     try:
#         ##common variables-------
#         #calibration-------------
#         cal_interp_order = 1
#         skytime = ctxcal['time_sky']
#         nrow_sky = ctxcal['nrow_sky']
#         tsystime = ctxcal['time_tsys']
#         nrow_tsys = ctxcal['nrow_tsys']
#         #offdata = libsakurapy.new_uninitialized_aligned_buffer(libsakurapy.TYPE_FLOAT, (nchan,))
#         #facdata = libsakurapy.new_uninitialized_aligned_buffer(libsakurapy.TYPE_FLOAT, (nchan,))
#         #mask--------------------
#         #mask_temp = libsakurapy.new_uninitialized_aligned_buffer(libsakurapy.TYPE_BOOL, (nchan,))
#         mask_bl = ctxbl['blmask']
#         
#         channel_id = ctxmc['channel_id']
#         edge_lower = ctxmc['edge_lower']
#         edge_upper = ctxmc['edge_upper']
#         ##clip--------------------
#         clip_lower = ctxmc['clip_lower']
#         clip_upper = ctxmc['clip_upper']
# 
#         ##convert to sakura-----------------
#         # input array shape: (npol, nrow)
#         # output tuple shape: (nrow, npol)
#         thedata = _casasakura.tosakura_float(in_data)
#         themask = _casasakura.tosakura_bool(in_mask)
#         
#         for ipol in pol_list:
#             ##convert to sakura-----------------
#             #data = _casasakura.tosakura_float(in_data[ipol])[0][0]
#             #mask = _casasakura.tosakura_bool(in_mask[ipol].flatten())[0][0]
#             data = thedata[0][ipol]
#             mask = themask[0][ipol]
# 
#             ##calibration-----------------------
#             #libsakurapy.interpolate_float_yaxis(libsakurapy.INTERPOLATION_METHOD_LINEAR, 
#             #                                    cal_interp_order, nrow_sky, skytime, 
#             #                                    nchan, ctxcal['sky'][ipol], 
#             #                                    1, datatime, offdata)
#             #libsakurapy.interpolate_float_yaxis(libsakurapy.INTERPOLATION_METHOD_LINEAR, 
#             #                                    cal_interp_order, nrow_tsys, tsystime, 
#             #                                    nchan, ctxcal['tsys'][ipol], 
#             #                                    1, datatime, facdata)
#             #result_cal = libsakurapy.apply_position_switch_calibration(nchan, facdata, 
#                                                                        nchan, data, offdata, data)
# 
#             ##masknaninf------------------------
#             #libsakurapy.set_false_float_if_nan_or_inf(nchan, data, mask_temp)
#             #libsakurapy.logical_and(nchan, mask_temp, mask, mask)
# 
#             ##maskedge--------------------------
#             #libsakurapy.set_true_int_in_ranges_exclusive(nchan, channel_id, 
#             #                                             1, edge_lower, edge_upper, mask_temp)
#             #libsakurapy.logical_and(nchan, mask_temp, mask, mask)
# 
#             ##baseline--------------------------
#             #libsakurapy.logical_and(nchan, mask, mask_bl, mask_temp)
#             #data = libsakurapy.subtract_baseline(nchan, data, mask_temp, 
#             #                                     ctxbl['context'], 
#             #                                     ctxbl['clip_threshold'], 
#             #                                     ctxbl['num_fitting_max'], 
#             #                                     True, mask_temp, data)
# 
#             ##clip------------------------------
#             #result_clip = libsakurapy.set_true_float_in_ranges_exclusive(nchan, data, 1, clip_lower, clip_upper, mask_temp)
#             #libsakurapy.logical_and(nchan, mask_temp, mask, mask)
#             
#             ##smooth----------------------------
#             #result_complement = libsakurapy.complement_masked_value_float(nchan, data, mask, data)
#             #result_smooth = libsakurapy.convolve1D(ctxsm, nchan, data, data)
# 
#             ##statistics------------------------
#             #stats = libsakurapy.compute_statistics(nchan, data, mask)
# 
#             ##convert to casa-------------------
#             #out_data[ipol] = _casasakura.tocasa_float(((data,),))
#             #out_mask[ipol] = _casasakura.tocasa_bool(((mask,),))
#             
#         ##convert to casa-------------------
#         #out_data = _casasakura.tocasa_float(thedata)
#         #out_mask = _casasakura.tocasa_bool(themask)
# 
#         # make sure that output arrays have same shape as inputs
#         #out_data = out_data.reshape(in_data.shape)
#         #out_mask = out_mask.reshape(in_mask.shape)
#         
#     except Exception as e:
#         print '[reducerecord]--'+e.message
#         raise
# 
#     #return (in_row, out_data, out_mask, in_time, 3.14)
# 
# def reducerecord2(record):
#     data, mask = tosakura(record[1], record[2])
#     data, mask = calibratedata(data, mask, record[3])
#     mask = masknanorinf(data, mask)
#     mask = maskedge(data, mask)
#     data, mask = baselinedata(data, mask)
#     mask = clipdata(data, mask)
#     data = smoothdata(data, mask)
#     stats = calcstats(data, mask)
#     data, flag = tocasa(data, mask)
#     yield (record[0], data, flag, record[3], stats)
#     
# def writechunk(table, results):
#     #print '                writechunk'
#     put = lambda row, col, val: table.putcell(col, row, val)
#     for record in results:
#         row = int(record[0])
#         data = record[1]
#         flag = record[2]
#         #print 'writing result to table %s at row %s...'%(table.name(), row)
#         put(row, 'CORRECTED_DATA', data)
#         put(row, 'FLAG', flag)

###
# def reducerecord_old(record):
#     print 'reducing row %s'%(record[0])
#     data, flag, stats = reducedata(record[0], record[1], record[2], record[3])
#     return (record[0], data, flag, record[3], stats)
# 
# def reducedata(row, data, flag, timestamp):
#     data[:] = float(row)
#     print 'reducing row %s...'%(row)
#     #mtpy.wait_for(5, 'row%s'%(row))
#     print 'done reducing row %s...'%(row)
#     return data, flag, {'statistics': data.real.mean()}

# BASELINE_TYPEMAP = {'poly': libsakurapy.BASELINE_TYPE_POLYNOMIAL,
#                     'chebyshev': libsakurapy.BASELINE_TYPE_CHEBYSHEV,
#                     'cspline': libsakurapy.BASELINE_TYPE_CUBIC_SPLINE,
#                     'sinusoid': libsakurapy.BASELINE_TYPE_SINUSOID}
# CONVOLVE1D_TYPEMAP = {'hanning': libsakurapy.CONVOLVE1D_KERNEL_TYPE_HANNING,
#                       'gaussian': libsakurapy.CONVOLVE1D_KERNEL_TYPE_GAUSSIAN,
#                       'boxcar': libsakurapy.CONVOLVE1D_KERNEL_TYPE_BOXCAR}
# CALIBRATION_TYPEMAP = {'nearest': libsakurapy.INTERPOLATION_METHOD_NEAREST,
#                        'linear': libsakurapy.INTERPOLATION_METHOD_LINEAR,
#                        'cspline': libsakurapy.INTERPOLATION_METHOD_SPLINE}

def sakura_typemap(typemap, key):
    try:
        return typemap[key.lower()]
    except KeyError as e:
        raise RuntimeError('Invalid type: %s'%(key))

def calibration_typemap(key):
    try:
        return CALIBRATION_TYPEMAP[key.lower()], 0
    except KeyError as e:
        if key.isdigit():
            return #libsakurapy.INTERPOLATION_METHOD_POLYNOMIAL, int(key)
        else:
            raise RuntimeError('Invalid type: %s'%(key))
        
def spw_id_map(vis):
    with opentable(os.path.join(vis, 'DATA_DESCRIPTION')) as tb:
        spwidmap = dict(((i,tb.getcell('SPECTRAL_WINDOW_ID',i)) for i in xrange(tb.nrows())))
    return spwidmap

def data_desc_id_map(vis):
    with opentable(os.path.join(vis, 'DATA_DESCRIPTION')) as tb:
        ddidmap = dict(((tb.getcell('SPECTRAL_WINDOW_ID',i),i) for i in xrange(tb.nrows())))
    return ddidmap
    
    
# def initcontext(vis, spw, antenna, gaintable, interp, spwmap,
#                 maskmode, thresh, avg_limit, edge, blmask,
#                 blfunc, order, npiece, applyfft, fftmethod,
#                 fftthresh, addwn, rejwn, clipthresh, clipniter,
#                 bloutput, blformat, clipminmax,
#                 kernel, kwidth, usefft, interpflag,
#                 statmask, stoutput, stformat):
#     ssd.initialize_sakura()
#     context_dict = {}
#     
#     # get nchan for each spw
#     #with opentable(os.path.join(vis, 'DATA_DESCRIPTION')) as tb:
#     #    ddidmap = dict(((tb.getcell('SPECTRAL_WINDOW_ID',i),i) for i in xrange(tb.nrows())))
#     ddidmap = data_desc_id_map(vis)
# 
#     # spw selection to index
#     if antenna is not None and len(antenna) > 0:
#         baseline = '%s&&&'%(antenna)
#     else:
#         baseline = ''
#     selection = ms.msseltoindex(vis=vis, spw=spw, baseline=baseline)
#     #print '-----selection-------------------------------------'
#     #print str(selection)
#     spwid_list = selection['spw']
#     if len(spwid_list) == 0:
#         """
#         with opentable(os.path.join(vis, 'DATA_DESCRIPTION')) as tb:
#             spwid_list = tb.getcol('SPECTRAL_WINDOW_ID')
#         """
#         #spwid_list = [3]
#         spwid_list = spwmap.keys()
#         for i in xrange(len(spwid_list)):
#             spwid_list[i] = int(spwid_list[i])
#     else:
#         spwid_list = list(set(spwid_list) & set(map(int, spwmap.keys())))
#     antennaid_list = selection['baselines'][:,0]
#     
#     # nchan
#     with opentable(os.path.join(vis, 'SPECTRAL_WINDOW')) as tb:
#         nchanmap = dict(((i,tb.getcell('NUM_CHAN',i)) for i in xrange(tb.nrows())))
#     
#     # create calibration context (base data for interpolation)
#     sky_tables = _select_sky_tables(gaintable)
#     tsys_tables = _select_tsys_tables(gaintable)
# 
#     for spwid in spwid_list:
#         nchan = nchanmap[spwid]
# 
#         # create calibration context
#         tsysspw = spwmap[str(spwid)] # interferometry style spwmap
#         calibration_context = create_calibration_context(vis,
#                                                          sky_tables,
#                                                          tsys_tables,
#                                                          spwid,
#                                                          tsysspw,
#                                                          antennaid_list,
#                                                          interp)
#         # create baseline context
#         if blmask is None or len(blmask) == 0:
#             idx = ms.msseltoindex(vis=vis,spw='%s:0~%s'%(spwid,nchan-1))
#         else:
#             idx = ms.msseltoindex(vis=vis,spw='%s:%s'%(spwid,blmask))
#         blmask_range = idx['channel'][:,1:3]
#         mask_bl_ = numpy.zeros(nchan, dtype=bool)
#         for r0, r1 in blmask_range:
#             mask_bl_[r0:r1+1] = True
# #         mask_bl = libsakurapy.new_aligned_buffer(libsakurapy.TYPE_BOOL, mask_bl_.tolist())
#         baseline_context = {}
#         baseline_context['blmask'] = mask_bl
#         baseline_context['clip_threshold'] = clipthresh
#         baseline_context['num_fitting_max'] = clipniter
#         baseline_type = sakura_typemap(BASELINE_TYPEMAP, blfunc)
# #         baseline_context['context'] = libsakurapy.create_baseline_context(baseline_type,
# #                                                                           order,
# #                                                                           nchan)
#         # create convolve1D context
#         convolve1d_type = sakura_typemap(CONVOLVE1D_TYPEMAP, kernel)
# #         convolve1d_context = libsakurapy.create_convolve1D_context(nchan,
# #                                                                    convolve1d_type,
# #                                                                    kwidth,
# #                                                                    usefft)
#         # create mask/clip context
#         maskclip_context = create_maskclip_context(nchan, edge, clipminmax)
# 
#         context_dict[spwid] = (calibration_context, baseline_context, convolve1d_context, maskclip_context)
# 
#     return context_dict
# 
# def create_calibration_context(vis, sky_tables, tsys_tables, spwid, tsysspw, antennaid_list, interp):
#     context = {}
#     # context must be prepared for each antenna
#     for antennaid in antennaid_list:
# 
#         # collect sky data for given antenna id and spw id
#         timestamp = None
#         data = None
#         for sky_table in sky_tables:
#             ddidmap = data_desc_id_map(sky_table)
#             if ddidmap.has_key(spwid):
#                 ddid = ddidmap[spwid]
#                 with opentable(sky_table) as tb:
#                     datacol = colname(tb)
#                     taql = 'DATA_DESC_ID==%s && ANTENNA1 == ANTENNA2 && ANTENNA1 == %s'%(ddid,antennaid)
#                     tsel = tb.query(taql)
#                     if tsel.nrows() > 0:
#                         if timestamp is None:
#                             timestamp = tsel.getcol('TIME')
#                         else:
#                             timestamp = numpy.concatenate([timestamp, tsel.getcol('TIME')], axis=0)
#                         if data is None:
#                             data = tsel.getcol(datacol)
#                         else:
#                             data = numpy.concatenate([data, tsel.getcol(datacol)], axis=1)
#                     tsel.close()
#                     
#         if timestamp is None or data is None:
#             raise RuntimeError('Empty sky data for antenna %s spw %s. Cannot proceed.'%(antennaid, spwid))
# 
#         # sort by time
#         sorted_time, sort_index = numpy.unique(timestamp, return_index=True)
#         #print 'sort_index', sort_index
#         #print 'data.shape', data.shape
#         sorted_data = data.take(sort_index, axis=2)
#         
#         # create aligned buffer for sky
#         # these are base data for interpolation so that it has to
#         # encapsulate flattened array for each polarization
#         time_sky = _casasakura.tosakura_double(sorted_time)[0][0]
#         npol, nchan, nrow_sky = sorted_data.shape
#         if datacol == 'DATA':
#             func = _casasakura.tosakura_complex
#         else:
#             func = _casasakura.tosakura_float
#         sky = tuple((func(sorted_data[i].flatten(order='F'))[0][0] for i in xrange(npol)))
# 
#         # collect Tsys data for given antenna id and spw id
#         timestamp = None
#         data = None
#         for tsys_table in tsys_tables:
#             with opentable(os.path.join(tsys_table, 'SYSCAL')) as tb:
#                 datacol = 'TSYS_SPECTRUM'
#                 if not datacol in tb.colnames():
#                     datacol = 'TSYS'
#                 tsel = tb.query('ANTENNA_ID==%s && SPECTRAL_WINDOW_ID==%s'%(antennaid,tsysspw))
# 		#print 'create_calibration_context[4]-antid='+str(antennaid)+',spwid='+str(tsysspw)+',nrows='+str(tsel.nrows())
#                 if tsel.nrows() > 0:
#                     if timestamp is None:
#                         timestamp = tsel.getcol('TIME')
#                     else:
#                         timestamp = numpy.concatenate([timestamp, tsel.getcol('TIME')], axis=0)
#                     if data is None:
#                         data = tsel.getcol(datacol)
#                     else:
#                         data = numpy.concatenate([data, tsel.getcol(datacol)], axis=1)
#                 tsel.close()
#         
#         if timestamp is None or data is None:
#             raise RuntimeError('Empty Tsys data for antenna %s spw %s. Cannot proceed.'%(antennaid, spwid))
#                     
#         # sort by time
#         sorted_time, sort_index = numpy.unique(timestamp, return_index=True)
#         sorted_data = data.take(sort_index, axis=2)
# 
#         # interpolate along spectral axis
#         # frequency labels are taken from vis
#         with opentable(os.path.join(vis, 'SPECTRAL_WINDOW')) as tb:
#             freq = tb.getcell('CHAN_FREQ', spwid)
#             freq_tsys = tb.getcell('CHAN_FREQ', tsysspw)
# 
#         # interpolation method
#         if len(interp[0]) > 0:
#             interpolation_method, poly_order = calibration_typemap(interp[0].split(',')[-1])
#         else:
# #             interpolation_method, poly_order = libsakurapy.INTERPOLATION_METHOD_LINEAR, 1
# 
#         # create aligned buffer for interpolation
#         def gen_interpolation():
# #             interpolated_freq = libsakurapy.new_aligned_buffer(libsakurapy.TYPE_DOUBLE, freq)
# #             base_freq = libsakurapy.new_aligned_buffer(libsakurapy.TYPE_DOUBLE, freq_tsys)
# #             base_freq = _casasakura.tosakura_double(freq_tsys)[0][0]
#             nchan = len(freq)
#             npol, nchan_base, nrow = sorted_data.shape
#             for ipol in xrange(npol):
#             	pass
# #                 base_data = _casasakura.tosakura_float(sorted_data[ipol].flatten(order='F'))[0][0]
# #                 interpolated_data = libsakurapy.new_uninitialized_aligned_buffer(libsakurapy.TYPE_FLOAT, (nchan * nrow,))
# #     
# #                 # perform interpolation
# #                 libsakurapy.interpolate_float_xaxis(interpolation_method,
# #                                                     poly_order,
# #                                                     nchan_base,
# #                                                     base_freq,
# #                                                     nrow,
# #                                                     base_data,
# #                                                     nchan,
# #                                                     interpolated_freq,
# #                                                     interpolated_data)
# #                 yield interpolated_data
# 
#         # data for context
# #         time_tsys = _casasakura.tosakura_double(sorted_time)[0][0]
#         tsys = tuple(gen_interpolation())
#         nrow_tsys = sorted_data.shape[2]
# 
#         context[antennaid] = {'nchan': nchan,
#                               'nrow_sky': nrow_sky,
#                               'nrow_tsys': nrow_tsys,
#                               'time_sky': time_sky,
#                               'time_tsys': time_tsys,
#                               'sky': sky,
#                               'tsys': tsys}
#     return context
#             
# def create_maskclip_context(nchan, edge, clipminmax):
#     context = {}
#     channel_id = libsakurapy.new_aligned_buffer(libsakurapy.TYPE_INT32, range(nchan))
#     if isinstance(edge, list) or isinstance(edge, tuple):
#         for i in xrange(len(edge)):
#             if not (isinstance(edge[i], float) or isinstance(edge[i], int) or isinstance(edge[i], tuple)):
#                 raise RuntimeError('Invalid type: %s'%(edge))
#         if len(edge) == 0:
#             edge_list = [0, 0]
#         elif len(edge) == 1:
#             edge_list = [int(edge[0]), int(edge[0])]
#         else:
#             edge_list = [int(edge[0]), int(edge[1])]
#     elif isinstance(edge, float) or isinstance(edge, int):
#         edge_list = [int(edge), int(edge)]
#     else:
#         raise RuntimeError('Invalid type: %s'%(edge))
#     edge_lower = libsakurapy.new_aligned_buffer(libsakurapy.TYPE_INT32, (edge_list[0]-1,))
#     edge_upper = libsakurapy.new_aligned_buffer(libsakurapy.TYPE_INT32, (nchan-edge_list[1],))
# 
#     if isinstance(clipminmax, list) or isinstance(clipminmax, tuple):
#         if len(clipminmax) == 0:
#             clipminmax = [0, 0]
#         elif len(clipminmax) == 1:
#             clipminmax = [-clipminmax[0], clipminmax[0]]
#         else:
#             clipminmax = clipminmax[0:2]
#     elif isinstance(clipminmax, float) or isinstance(clipminmax, int):
#         clipminmax = [float(-abs(clipminmax)), float(abs(clipminmax))]
#     else:
#         raise RuntimeError('Invalid type: %s'%(clipminmax))
#     clip_lower = libsakurapy.new_aligned_buffer(libsakurapy.TYPE_FLOAT, (min(clipminmax),))
#     clip_upper = libsakurapy.new_aligned_buffer(libsakurapy.TYPE_FLOAT, (max(clipminmax),))
# 
#     context = {'channel_id': channel_id,
#                'edge_lower': edge_lower,
#                'edge_upper': edge_upper,
#                'clip_lower': clip_lower,
#                'clip_upper': clip_upper}
#     return context

# def _select_sky_tables(gaintable):
#     return list(_select_match(gaintable, 'sky'))
# 
# def _select_tsys_tables(gaintable):
#     return list(_select_match(gaintable, 'tsys'))
# 
# def _select_match(gaintable, tabletype):
#     pattern = '_%s(\.ms/?)?$'%(tabletype.lower())
#     for caltable in gaintable:
#         if re.search(pattern, caltable):
#             yield caltable
# 
# def colname(tb):
#     colnames = tb.colnames()
#     if 'FLOAT_DATA' in colnames:
#         return 'FLOAT_DATA'
#     else:
#         return 'DATA'
# 
# def add_corrected_data(table):
#     with opentable(table) as tb:
#         colnames = tb.colnames()
#         if 'CORRECTED_DATA' not in colnames:
#             if 'DATA' in colnames:
#                 desc = tb.getcoldesc('DATA')
#             elif 'FLOAT_DATA' in colnames:
#                 desc = tb.getcoldesc('FLOAT_DATA')
#                 desc['valueType'] = 'complex'
#             desc['comment'] = 'corrected data'
#             tb.addcols({'CORRECTED_DATA': desc})
