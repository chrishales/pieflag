from taskinit import *
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Copyright (c) 2014, Christopher A. Hales
# All rights reserved.
#
# BSD 3-Clause Licence
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The NAMES OF ITS CONTRIBUTORS may not be used to endorse or promote
#       products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL C. A. HALES BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# HISTORY:
#   1.0       2005  Initial version by Enno Middelberg, designed
#                   for use with customized UVLIST in MIRIAD
#   1.1    Jan2006  Various upgrades by Enno Middelberg
#   2.0  31Oct2014  Release of updated and CASA-compatible version
#                   written by Christopher A. Hales
#   2.1  26Nov2014  Fixed subscan bug (only operate on '0') and
#                   logger default value printout
#

# See additional information in pieflag function

def pieflag_getflagstats(vis,field,spw,npol):
    casalog.filter('WARN')
    af.open(msname=vis)
    af.selectdata(field=str(field),spw=str(spw))
    ag0={'mode':'summary','action':'calculate'}
    af.parseagentparameters(ag0)
    af.init()
    temp=af.run(writeflags=False)
    af.done
    casalog.filter('INFO')
    RR=temp['report0']['correlation']['RR']['flagged'] / temp['report0']['correlation']['RR']['total'] * 100
    LL=temp['report0']['correlation']['LL']['flagged'] / temp['report0']['correlation']['LL']['total'] * 100
    comb=temp['report0']['flagged'] / temp['report0']['total'] * 100
    flagstats=np.array([RR,LL,comb])
    if npol == 4:
        RL=temp['report0']['correlation']['RL']['flagged'] / temp['report0']['correlation']['RL']['total'] * 100
        LR=temp['report0']['correlation']['LR']['flagged'] / temp['report0']['correlation']['LR']['total'] * 100
        flagstats=np.append(flagstats,[RL,LR])
    
    return flagstats

def pieflag_flag(vis,field,
                 vtbleLIST,inttime,nant,bL,nb,
                 ddid,spw,refchan,nchan,npol,
                 fitorderLIST,sefdLIST,
                 staticflag,madmax,binsamples,
                 dynamicflag,chunktime,stdmax,maxoffset,
                 extendflag,boxtime,boxthresh):
    
    # Go through each baseline, spw, channel, and polarization and compare to reference channel
    # while accounting for a spectral fit and the SEFD.
    # Perform static, dynamic, and extend operations if requested
    
    casalog.post('    status: 0% complete (updates delivered every 10%)')
    
    vtble = np.array(vtbleLIST)
    sefd = np.array(sefdLIST)
    fitorder = np.array(fitorderLIST)
    
    nspw=len(spw)
    nbaselines=nant*(nant-1)/2
    ant1 = 0
    ant2 = 1
    pSTR = ['RR']
    if npol == 2:
        pSTR.append('LL')
    elif npol == 4:
        pSTR.append('RL')
        pSTR.append('LR')
        pSTR.append('LL')
    
    # dim0 --> npol=2: 0=RR, 1=LL
    #          npol=4: 0=RR, 1=RL, 2=LR, 3=LL
    specfitcoeffS=np.zeros((npol,max(fitorder)+1))
    
    # rc  = reference channel
    # rcx = frequency in Hz for static flagging
    rcx=np.zeros(nspw)
    for i in range(nspw):
        rcx[i] = vtble[refchan[i]][spw[i]]
    
    # S = static
    # Srcy: dim2=(median visibility amplitude, median absolute deviation)
    Srcy=np.zeros((nspw,npol,2))
    
    if extendflag:
        kernellen = int(boxtime/inttime)
        #kernel = np.ones(kernellen)
    
    tb.open(vis)
    ms.open(vis,nomodify=False)
    printupdate=np.ones(9).astype(bool)
    printcounter=1
    checkprint=True
    for b in range(nbaselines):
        if b >= bL and b < bL+nb:
            if checkprint:
                if printupdate[printcounter-1] and b-bL+1>nb/10*printcounter:
                    casalog.post('    status: '+str(10*printcounter)+'% complete')
                    printupdate[printcounter-1]=False
                    printcounter+=1
                    if printcounter > 9:
                        checkprint=False
            
            # get reference channel median and MAD for static flagging
            validspw = np.zeros((npol,nspw))
            for s in range(nspw):
                for p in range(npol):
                    tempstr1 = '([select from '+vis+' where ANTENNA1=='+str(ant1)+' && ANTENNA2=='+str(ant2)+\
                               ' && FIELD_ID=='+str(field)+' && DATA_DESC_ID=='+str(ddid[s])+\
                               ' && FLAG_ROW==False && FLAG['+str(p)+','+str(refchan[s])+']==False giving '
                    #           ' && WEIGHT['+str(p)+']>0 giving '
                    tempstr2 = '[abs(CORRECTED_DATA['+str(p)+','+str(refchan[s])+'])]])'
                    tempval = tb.calc('count'+tempstr1+tempstr2)[0]
                    if tempval > 0:
                        validspw[p][s] = 1
                        if staticflag:
                            Srcy[s][p][0] = tb.calc('median'+tempstr1+tempstr2)[0]
                            tempstr3 = '[abs(abs(CORRECTED_DATA['+str(p)+','+str(refchan[s])+'])-'+\
                                       str(Srcy[s][p][0])+')]])'
                            Srcy[s][p][1] = tb.calc('median'+tempstr1+tempstr3)[0]
                    else:
                        # If the reference channel for any one polarization isn't present,
                        # flag all data on this baseline in this spw.
                        # You won't be able to do static or dynamic flagging (nor extend flagging as a result).
                        # This part of the loop shouldn't get activated much on unflagged data because the
                        # user should have picked a suitable reference channel in each spw.
                        validspw[0][s] = 0
                        casalog.filter('WARN')
                        ms.reset()
                        ms.msselect({'field':str(field),'baseline':str(ant1)+'&&'+str(ant2),'spw':str(spw[s])})
                        # for some reason I can't do the following with flag_row? That or plotms is broken...
                        tempflag = ms.getdata('flag')
                        tempflag['flag'][:]=True
                        ms.putdata(tempflag)
                        casalog.filter('INFO')
                        break
            
            # get static spectral fits for each polarization
            if staticflag:
                tempfitorderS = np.copy(fitorder)
                for p in range(npol):
                    # check that there are enough spw's to fit the requested spectral order
                    if sum(validspw[p]) > 0:
                        if tempfitorderS[p] > sum(validspw[p])-1:
                            if sum(validspw[p]) == 2:
                                tempfitorderS[p] = 1
                            else:
                                tempfitorderS[p] = 0
                            
                            casalog.post('*** WARNING: staticflag fitorder for baseline ant1='+str(ant1)+' ant2='+str(ant2)+\
                                         ' pol='+pSTR[p]+' has been reduced to '+str(int(tempfitorderS[p])),'WARN')
                        
                        # use MAD to weight the points
                        # (not mathematically correct, should be standard error, but OK approximation)
                        specfitcoeffS[p,0:tempfitorderS[p]+1] = np.polyfit(np.log10(rcx[validspw[p]>0]),\
                                                                np.log10(Srcy[0:,p,0][validspw[p]>0]),\
                                                                tempfitorderS[p],w=1.0/np.log10(Srcy[0:,p,1][validspw[p]>0]))
            
            if dynamicflag:
                # Don't assume that the same number of integrations (dump times) are present in each spw.
                # This requirement makes the code messy
                casalog.filter('WARN')
                ms.reset()
                ms.msselect({'field':str(field),'baseline':str(ant1)+'&&'+str(ant2)})
                ms.iterinit(interval=chunktime,columns=['TIME'],adddefaultsortcolumns=False)
                # get number of chunks and initialize arrays
                ms.iterorigin()
                moretodo=True
                nchunks = 0
                while moretodo:
                    nchunks += 1
                    moretodo = ms.iternext()
                
                # start and end timestamp for each chunk
                timestamps = np.zeros((nchunks,2))
                
                # D = dynamic
                # dim3 (per chunk) --> 0=reference channel median, 1=reference channel standard deviation
                Drcy=np.zeros((nspw,npol,nchunks,2))
                
                validspwD = np.zeros((npol,nchunks,nspw))
                
                ms.iterorigin()
                moretodo=True
                chunk = 0
                while moretodo:
                    tempflagD = ms.getdata('flag')['flag']
                    tempdataD = abs(ms.getdata('corrected_data')['corrected_data'])
                    tempddidD = ms.getdata('data_desc_id')['data_desc_id']
                    for s in range(nspw):
                        for p in range(npol):
                            # messy...
                            messydata1 = tempdataD[p,refchan[s]][tempflagD[p,refchan[s]]==False]
                            if len(messydata1) > 0:
                                messyddid  = tempddidD[tempflagD[p,refchan[s]]==False]
                                messydata2 = messydata1[messyddid==ddid[s]]
                                if len(messydata2) > 0:
                                    validspwD[p,chunk,s] = 1
                                    Drcy[s,p,chunk,0] = np.median(messydata2)
                                    Drcy[s,p,chunk,1] = np.std(messydata2)
                    
                    # Get start and end timestamps so the data can be matched up later.
                    # The overall timespan reported here will be equal to or greater
                    # than the timespan reported below when ms.getdata is run on an
                    # individual spw, because we need to account for the possible
                    # presence of some spw's with less integrations. Messy...
                    temptimeD = ms.getdata('time')['time']
                    timestamps[chunk,0] = min(temptimeD)
                    timestamps[chunk,1] = max(temptimeD)
                    
                    chunk += 1
                    moretodo = ms.iternext()
                
                # get dynamic spectral fits for each polarization
                tempfitorderD = np.zeros((nchunks,len(fitorder)))
                for i in range(len(fitorder)):
                    tempfitorderD[:,i] = fitorder[i]
                
                # dim0 --> npol=2: 0=RR, 1=LL
                #          npol=4: 0=RR, 1=RL, 2=LR, 3=LL
                specfitcoeffD=np.zeros((npol,nchunks,max(fitorder)+1))
                
                ms.iterorigin()
                moretodo=True
                chunk = 0
                while moretodo:
                    for p in range(npol):
                        # check that there are enough spw's to fit the requested spectral order
                        if sum(validspwD[p,chunk]) > 0:
                            if tempfitorderD[chunk,p] > sum(validspwD[p,chunk])-1:
                                if sum(validspwD[p,chunk]) == 2:
                                    tempfitorderD[chunk,p] = 1
                                else:
                                    tempfitorderD[chunk,p] = 0
                                
                                # native time is MJD seconds
                                t1=qa.time(qa.quantity(timestamps[chunk,0],'s'),form='ymd')[0]
                                t2=qa.time(qa.quantity(timestamps[chunk,1],'s'),form='d')[0]
                                casalog.post('*** WARNING: dynamicflag fitorder for baseline ant1='+str(ant1)+' ant2='+str(ant2)+\
                                             ' pol='+pSTR[p]+' time='+t1+'-'+t2+\
                                             ' has been reduced to '+str(int(tempfitorderD[chunk,p])),'WARN')
                            
                            specfitcoeffD[p,chunk,0:tempfitorderD[chunk,p]+1] = \
                                np.polyfit(np.log10(rcx[validspwD[p,chunk]>0]),np.log10(Drcy[0:,p,chunk,0][validspwD[p,chunk]>0]),\
                                tempfitorderD[chunk,p],w=1.0/np.log10(Drcy[0:,p,chunk,1][validspwD[p,chunk]>0]))
                    
                    chunk += 1
                    moretodo = ms.iternext()
                
                casalog.filter('INFO')
            
            for s in range(nspw):
                if validspw[0,s] > 0:
                    casalog.filter('WARN')
                    ms.reset()
                    ms.msselect({'field':str(field),'baseline':str(ant1)+'&&'+str(ant2),'spw':str(spw[s])})
                    # get data for this spw, accounting for existing flags
                    tempflag = ms.getdata('flag')
                    tempdata = abs(ms.getdata('corrected_data')['corrected_data'])
                    tempflagpf = np.zeros(tempdata.shape)
                    temptime = ms.getdata('time')['time']
                    casalog.filter('INFO')
                    
                    if staticflag:
                        windowtime = binsamples * inttime
                        window = []
                        casalog.filter('WARN')
                        ms.iterinit(interval=windowtime)
                        ms.iterorigin()
                        # get number of time steps
                        moretodo=True
                        while moretodo:
                            # select from dummy column with small data size, eg int 'antenna1'
                            # (could also have used float 'time'...)
                            window.append(len(ms.getdata('antenna1')['antenna1']))
                            moretodo = ms.iternext()
                        
                        casalog.filter('INFO')
                        for f in range(nchan):
                            # this shouldn't matter, but enforce that flagging
                            # doesn't take place on the reference channel
                            if f == refchan[s]:
                                continue
                            
                            for p in range(npol):
                                if tempfitorderS[p] > 0:
                                    specfit = 10.0**(np.poly1d(specfitcoeffS[p,0:tempfitorderS[p]+1])(np.log10(vtble[f][spw[s]])))
                                else:
                                    specfit = Srcy[s][p][0]
                                
                                # difference to median of reference channel, accounting for spectrum and sefd
                                tempdatachan = np.multiply(abs((tempdata[p][f]-specfit)/sefd[s][f]),np.invert(tempflag['flag'][p][f]))
                                
                                tempbad = np.zeros(tempdatachan.shape)
                                tempbad[tempdatachan>=Srcy[s,p,1]*madmax] = 1
                                tempbad[tempdatachan>=Srcy[s,p,1]*madmax*2] += 1
                                
                                # iterate in units of binsamples*inttime
                                # flag entire window if sum of badness values >=2
                                # if flagging needs to take place in one polarization, just flag them all
                                j=0
                                for w in window:
                                    if sum(tempbad[j:j+w]) >= 2:
                                        tempflagpf[0:npol,f,j:j+w] = 1
					tempflag['flag'][0:npol,f,j:j+w] = True
                                    
                                    j+=w
                    
                    if dynamicflag:
                        for chunk in range(nchunks):
                            # calculate index range that matches up with timestamps
                            tL = np.where(temptime==timestamps[chunk,0])[0][0]
                            tU = np.where(temptime==timestamps[chunk,1])[0][0]
                            for p in range(npol):
                                if validspwD[p,chunk,s] == 1:
                                    for f in range(nchan):
                                        # this shouldn't matter, but enforce that flagging
                                        # doesn't take place on the reference channel
                                        if f == refchan[s]:
                                            continue
                                        
                                        if tempfitorderD[chunk,p] > 0:
                                            specfit = 10.0**(np.poly1d(specfitcoeffD[p,chunk,0:tempfitorderD[chunk,p]+1])(np.log10(vtble[f][spw[s]])))
                                        else:
                                            specfit = Drcy[s,p,chunk,0]
                                        
                                        # get channel data
                                        tempdatachan = np.multiply(tempdata[p,f,tL:tU+1],np.invert(tempflag['flag'][p,f,tL:tU+1]))
                                        tempstd = np.std(tempdatachan[tempdatachan>0])/sefd[s][f]
                                        
                                        if (tempstd >= stdmax*Drcy[s,p,chunk,1]) or \
                                           (abs(np.median(tempdatachan[tempdatachan>0])-specfit) >= maxoffset*tempstd):
                                            # if flagging needs to take place in one polarization, just flag them all
                                            tempflagpf[0:npol,f,tL:tU+1] = 2
                                            tempflag['flag'][0:npol,f,tL:tU+1] = True
                                    
                                else:
                                    # If the reference channel for any one polarization isn't present,
                                    # flag all data in this chunk on this baseline in this spw.
                                    # This part of the loop shouldn't get activated much on unflagged data because the
                                    # user should have picked a suitable reference channel in each spw.
                                    tempflag['flag'][0:npol,0:nchan,tL:tU+1]=True
                                    break
                    
                    if extendflag:
                        tempscanfull = ms.getscansummary()
                        tempscankeys = map(int,tempscanfull.keys())
                        tempscankeys.sort()
                        tempscan = []
                        for j in tempscankeys:
                            tempscan.append(tempscanfull[str(j)]['0']['nRow'])
                        
                        # only consider flags that have been set by pieflag, not pre-existing flags
                        j=0
                        for w in tempscan:
                            for f in range(nchan):
                                if f == refchan[s]:
                                    continue
                                
                                for p in range(npol):
                                    # convolve if kernel is smaller than scan length
                                    # otherwise, just use fraction of flagged values in scan
                                    if w > kernellen:
                                        #tempcon = np.convolve(tempflag['flag'][p][f][j:j+w],kernel,'valid')
                                        #tempcon = np.convolve(tempflagchan[j:j+w],kernel,'valid')
                                        for k in range(w-kernellen+1):
                                            #tempfrac = float(sum(tempflag['flag'][p][f][j+k:j+k+kernellen]))/float(kernellen)
                                            tempfrac = float(sum(tempflagpf[p,f,j+k:j+k+kernellen]))/float(kernellen)
                                            if tempfrac > boxthresh/100.0:
                                                tempflag['flag'][0:npol,f,j+k:j+k+kernellen] = True
                                    else:
                                        #tempfrac=float(sum(tempflag['flag'][p][f][j:j+w]))/float(w)
                                        tempfrac=float(sum(tempflagpf[p,f,j:j+w]))/float(w)
                                        if tempfrac > boxthresh/100.0:
                                            tempflag['flag'][0:npol,f,j:j+w] = True
                            
                            j+=w
                    
                    ms.putdata(tempflag)
        
        ant2 += 1
        if ant2 > nant-1:
            ant1 += 1
            ant2 = ant1 + 1
    
    ms.close()
    tb.close()
    casalog.post('    status: 100% complete')
    return

def pieflag(vis,
            field,          # data selection parameters
            refchanfile,
            fitorder_RR_LL,
            fitorder_RL_LR,
            scalethresh,
            SEFDfile,       # scalethresh parameter
            plotSEFD,
            dynamicflag,
            chunktime,      # dynamicflag parameters
            stdmax,
            maxoffset,
            staticflag,
            madmax,         # staticflag parameter
            binsamples,
            extendflag,
            boxtime,        # extendflag parameters
            boxthresh):

    #
    # Task pieflag
    #    Flags bad data by comparing with clean channels in bandpass-calibrated data.
    #
    #    Original reference: E. Middelberg, 2006, PASA, 23, 64
    #    Rewritten for use in CASA and updated to account for
    #    wideband and SEFD effects by Christopher A. Hales 2014.
    #    Thanks to Kumar Golap, Justo Gonzalez, Jeff Kern,
    #    Urvashi Rau, Sanjay Bhatnagar, and of course Enno Middelberg
    #    for expert advice. Thanks to Emmanuel Momjian for providing
    #    Jansky VLA SEFD data for L and X bands (EVLA Memos 152 and 166)
    #    and to Bryan Butler for providing access to all other bands
    #    from the Jansky VLA Exposure Calculator.
    #
    #    Version 2.1 released 26 November 2014
    #    Tested with CASA Version 4.3.0 using Jansky VLA data
    #    Available at: http://github.com/chrishales/pieflag
    #
    #    Reference for this version:
    #    C. A. Hales, E. Middelberg, 2014, Astrophysics Source Code Library, 1408.014
    #    http://adsabs.harvard.edu/abs/2014ascl.soft08014H
    #
    
    startTime = time.time()
    casalog.origin('pieflag')
    
    if (not staticflag) and (not dynamicflag):
        casalog.post('*** ERROR: You need to select static or dynamic flagging.', 'ERROR')
        casalog.post('*** ERROR: Exiting pieflag.', 'ERROR')
        return
    
    ms.open(vis)
    vis=ms.name()
    ms.close()
    
    # load in reference channel details
    # OK, there are probably more elegant ways
    # of implementing the following code...meh
    refchandict=json.load(open(refchanfile))
    spw=[]
    for i in refchandict.keys():
        spw.append(int(i))
    
    nspw=len(spw)
    # json doesn't seem to load in the spw order properly
    # The user might not have entered spw's in order either
    # so perform sort just in case
    # note: no need to perform sort on the string versions
    spw.sort()
    # now get reference channels in corresponding sorted order
    refchan=[]
    for i in range(nspw):
        refchan.append(refchandict[str(spw[i])])
    
    # open MS and select relevant data
    ms.open(vis)
    ms.msselect({'field':str(field)})
    
    # get integration time
    scan_summary = ms.getscansummary()
    ms.close()
    scan_list = []
    for scan in scan_summary:
        if scan_summary[scan]['0']['FieldId'] == field:
            scan_list.append(int(scan))
    
    inttime=scan_summary[str(scan_list[0])]['0']['IntegrationTime']
    for i in range(len(scan_list)):
        if inttime != scan_summary[str(scan_list[i])]['0']['IntegrationTime']:
            casalog.post('*** ERROR: Bummer, pieflag is not set up to handle '+\
                              'changing integration times throughout your MS.', 'ERROR')
            casalog.post('*** ERROR: Exiting pieflag.','ERROR')
            return
    
    # get number of baselines
    tb.open(vis+'/ANTENNA')
    atble=tb.getcol('NAME')
    tb.close
    nant=atble.shape[0]
    nbaselines=nant*(nant-1)/2
    
    # channel to frequency (Hz) conversion
    tb.open(vis+'/SPECTRAL_WINDOW')
    vtble=tb.getcol('CHAN_FREQ')
    tb.close
    # vtble format is vtble[channel][spw]
    # assume each spw has the same number of channels
    nchan=vtble.shape[0]
    
    # get number of polarizations, assume they don't change throughout observation
    # get details from the first user-selected spw within the first scan on target field
    # note: I won't assume that spw specifies data_desc_id in the main table, even
    #       though in most cases it probably does. Probably overkill given the lack
    #       of checks done elsewhere in this code...
    tb.open(vis+'/DATA_DESCRIPTION')
    temptb=tb.query('SPECTRAL_WINDOW_ID='+str(spw[0]))
    # while here, get the data_desc_id values that pair with spw number
    tempddid=tb.getcol('SPECTRAL_WINDOW_ID').tolist()
    ddid=[]
    for s in range(nspw):
        ddid.append(tempddid.index(spw[s]))
    
    tb.close
    polid=temptb.getcell('POLARIZATION_ID')
    tb.open(vis+'/POLARIZATION')
    npol=tb.getcell('NUM_CORR',polid)
    tb.close
    
    if not (npol == 2 or npol == 4):
        casalog.post('*** ERROR: You data contains '+str(npol)+' polarization products.','ERROR')
        casalog.post('*** ERROR: pieflag can only handle 2 (RR/LL) or 4 (RR/RL/LR/LL). Exiting.','ERROR')
        return
    
    casalog.post('--> Some details about your data:')
    casalog.post('    integration time = '+str(inttime)+' sec')
    casalog.post('    number of baselines = '+str(nbaselines))
    casalog.post('    spectral windows to process = '+str(spw))
    casalog.post('    number of channels per spectral window = '+str(nchan))
    casalog.post('    number of polarization products to process = '+str(npol))
    casalog.post('--> Statistics of pre-existing flags:')
    flag0 = np.zeros((nspw,npol+1))
    for i in range(nspw):
        flag0[i] = pieflag_getflagstats(vis,field,spw[i],npol)
        RRs="{:.1f}".format(flag0[i][0])
        LLs="{:.1f}".format(flag0[i][1])
        comb="{:.1f}".format(flag0[i][2])
        if npol == 2:
            outstr='    flagged data in spw='+str(spw[i])+':  RR='+RRs+'%  LL='+LLs+'%  total='+comb+'%'
        else:
            RLs="{:.1f}".format(flag0[i][3])
            LRs="{:.1f}".format(flag0[i][4])
            outstr='    flagged data in spw='+str(spw[i])+':  RR='+RRs+'%  RL='+RLs+'%  LR='+LRs+'%  LL='+LLs+'%  total='+comb+'%'
        
        casalog.post(outstr)
    
    # Check there are enough spectral windows to perform the fitting later on. If not, lower the order.
    if fitorder_RR_LL > nspw-1:
        if fitorder_RR_LL == 2:
            casalog.post('*** WARNING: pieflag needs at least 3 spectral windows to fit for RR or LL spectral curvature.','WARN')
        else:
            casalog.post('*** WARNING: pieflag needs at least 2 spectral windows to fit for RR or LL spectral index.','WARN')
        
        if nspw == 2:
            fitorder_RR_LL=1
        else:
            fitorder_RR_LL=0
        
        casalog.post('*** WARNING: fitorder_RR_LL has been reduced to '+str(int(fitorder_RR_LL))+ ' and','WARN')
        casalog.post('***          may be reduced further for some baselines if the','WARN')
        casalog.post('***          reference channel isn\'t available in all selected spw\'s.','WARN')
    
    if npol == 2:
        fitorder    = np.zeros(2)
        fitorder[0] = fitorder_RR_LL
        fitorder[1] = fitorder_RR_LL
    elif npol == 4:
        if fitorder_RL_LR > nspw-1:
            if fitorder_RL_LR == 2:
                casalog.post('*** WARNING: pieflag needs at least 3 spectral windows to fit for RL or LR spectral curvature.','WARN')
            else:
                casalog.post('*** WARNING: pieflag needs at least 2 spectral windows to fit for RL or LR spectral index.','WARN')
            
            if nspw == 2:
                fitorder_RL_LR=1
            else:
                fitorder_RL_LR=0
            
            casalog.post('*** WARNING: fitorder_RL_LR has been reduced to '+str(int(fitorder_RL_LR))+' and','WARN')
            casalog.post('***          may be reduced further for some baselines if the','WARN')
            casalog.post('***          reference channel isn\'t available in all selected spw\'s.','WARN')
        
        fitorder    = np.zeros(4)
        fitorder[0] = fitorder_RR_LL
        fitorder[1] = fitorder_RL_LR
        fitorder[2] = fitorder_RL_LR
        fitorder[3] = fitorder_RR_LL
    
    if scalethresh:
        # read in SEFD data and interpolate to get values at our channel frequencies
        casalog.post('--> Reading in SEFD and interpolating at channel frequencies...')
        sefdRAW=np.loadtxt(SEFDfile)
        sefd=np.zeros((nspw,nchan))
        if not np.all(np.diff(sefdRAW[:,0]) >= 0):
            casalog.post('*** ERROR: Your SEFD file must be in order of increasing frequency.','ERROR')
            casalog.post('*** ERROR: Exiting pieflag.','ERROR')
            return
        
        sefdINTERP = interp1d(sefdRAW[:,0],sefdRAW[:,1])
        for i in range(nspw):
            sefdREFCHAN = sefdINTERP(vtble[refchan[i]][spw[i]])
            for j in range(nchan):
                # values in each spectral window will be relative to the reference channel value
                sefd[i][j] = sefdINTERP(vtble[j][spw[i]]) / sefdREFCHAN
        
        if plotSEFD:
            # clunky, but works, meh...
            sefdPLOT=np.zeros((nspw*nchan,3))
            k=0
            for i in range(nspw):
                sefdREFCHAN = sefdINTERP(vtble[refchan[i]][spw[i]])
                for j in range(nchan):
                    sefdPLOT[k][0] = vtble[j][spw[i]]/1.0e9
                    sefdPLOT[k][1] = sefd[i][j] * sefdREFCHAN
                    sefdPLOT[k][2] = sefd[i][j]
                    k += 1
            
            f, (ax1, ax2) = plt.subplots(2,sharex=True)
            ax1.plot(sefdRAW[:,0]/1.0e9,sefdRAW[:,1],'b-',sefdPLOT[:,0],sefdPLOT[:,1],'r.',markersize=10)
            ax2.plot([sefdRAW[0,0]/1.0e9,sefdRAW[len(sefdRAW[:,0])-1,0]/1.0e9],[1.,1.],'c-',sefdPLOT[:,0],sefdPLOT[:,2],'r.',markersize=10)
            f.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
            ax1.set_title('relative sensitivity assumed across your band,\nnormalized to the reference channel in each spw')
            ax1.legend(['raw input','interpolated'])
            ax1.set_ylabel('SEFD (arbitrary units)')
            ax2.set_xlabel('frequency (GHz)')
            ax2.set_ylabel('SEFD (normalized units per spw)')
    else:
        sefd=np.ones((nspw,nchan))
    
    if not staticflag:
        madmax = 0
        binsamples = 0
    
    if not dynamicflag:
        chunktime = 0
        stdmax = 0
        maxoffset = 0
    
    if not extendflag:
        boxtime = 0
        boxthresh = 0
    
    casalog.post('--> pieflag will now flag your data...')
    
    pieflag_flag(vis,field,vtble.tolist(),inttime,nant,0,nbaselines,
                 ddid,spw,refchan,nchan,npol,
                 fitorder.tolist(),sefd.tolist(),
                 staticflag,madmax,binsamples,
                 dynamicflag,chunktime,stdmax,maxoffset,
                 extendflag,boxtime,boxthresh)
    
    # show updated flagging statistics
    casalog.post('--> Statistics of final flags (including pre-existing):')
    flag1 = np.zeros((nspw,npol+1))
    for i in range(nspw):
        flag1[i] = pieflag_getflagstats(vis,field,spw[i],npol)
        RRs="{:.1f}".format(flag1[i][0])
        LLs="{:.1f}".format(flag1[i][1])
        comb="{:.1f}".format(flag1[i][2])
        if npol == 2:
            outstr='    flagged data in spw='+str(spw[i])+':  RR='+RRs+'%  LL='+LLs+'%  total='+comb+'%'
        else:
            RLs="{:.1f}".format(flag1[i][3])
            LRs="{:.1f}".format(flag1[i][4])
            outstr='    flagged data in spw='+str(spw[i])+':  RR='+RRs+'%  RL='+RLs+'%  LR='+LRs+'%  LL='+LLs+'%  total='+comb+'%'
        
        casalog.post(outstr)
    
    casalog.post('--> Statistics of pieflag flags (excluding pre-existing):')
    for i in range(nspw):
        RRs="{:.1f}".format(flag1[i][0]-flag0[i][0])
        LLs="{:.1f}".format(flag1[i][1]-flag0[i][1])
        comb="{:.1f}".format(flag1[i][2]-flag0[i][2])
        if npol == 2:
            outstr='    data flagged in spw='+str(spw[i])+':  RR='+RRs+'%  LL='+LLs+'%  total='+comb+'%'
        else:
            RLs="{:.1f}".format(flag1[i][3]-flag0[i][3])
            LRs="{:.1f}".format(flag1[i][4]-flag0[i][4])
            outstr='    data flagged in spw='+str(spw[i])+':  RR='+RRs+'%  RL='+RLs+'%  LR='+LRs+'%  LL='+LLs+'%  total='+comb+'%'
        
        casalog.post(outstr)
    
    t=time.time()-startTime
    casalog.post('--> pieflag run time:  '+str(int(t//3600))+' hours  '+\
                 str(int(t%3600//60))+' minutes  '+str(int(t%60))+' seconds')
