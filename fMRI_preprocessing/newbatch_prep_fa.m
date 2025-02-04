%run preprocessing batch
clear;
clc;

[num,txt,raw] = xlsread('/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/fileDetection.xls');

subjs = raw(2:end,1);%242 failed, 231x excluded 
%selection = raw(2:end,4);% this selection for head motion
behavior = raw(2:end,5);

for i = 1:length(subjs)
    subj = subjs{i};
    
    if behavior{i} == 1
        anatName = ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/',subj,'/anat/',subj,'_ana.nii,1'];

        %input PEI data(first 8 volumes were removed)
        i=1;
        for j = 1:160
            funcName{i} =  ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/',subj,'/fa/func.nii,',num2str(j)];
            i = i+1;
        end

        %funcName = funcName(5:164);

        funcPath = ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/',subj,'/fa/'];

        matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_named_dir.name = 'subject directory';
        matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_named_dir.dirs = {{'/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm'}};
        matlabbatch{2}.cfg_basicio.file_dir.dir_ops.cfg_cd.dir(1) = cfg_dep('Named Directory Selector: subject directory(1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dirs', '{}',{1}));
        %%realign
        matlabbatch{3}.spm.spatial.realign.estwrite.data = {funcName'};
        %%
        matlabbatch{3}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
        matlabbatch{3}.spm.spatial.realign.estwrite.eoptions.sep = 4;
        matlabbatch{3}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
        matlabbatch{3}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
        matlabbatch{3}.spm.spatial.realign.estwrite.eoptions.interp = 2;
        matlabbatch{3}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
        matlabbatch{3}.spm.spatial.realign.estwrite.eoptions.weight = '';
        matlabbatch{3}.spm.spatial.realign.estwrite.roptions.which = [2 1];
        matlabbatch{3}.spm.spatial.realign.estwrite.roptions.interp = 4;
        matlabbatch{3}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
        matlabbatch{3}.spm.spatial.realign.estwrite.roptions.mask = 1;
        matlabbatch{3}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
        %slice timing
        matlabbatch{4}.spm.temporal.st.scans{1}(1) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 1)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','rfiles'));
        matlabbatch{4}.spm.temporal.st.nslices = 30;
        matlabbatch{4}.spm.temporal.st.tr = 2;
        matlabbatch{4}.spm.temporal.st.ta = 1.95;
        matlabbatch{4}.spm.temporal.st.so = [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30];
        matlabbatch{4}.spm.temporal.st.refslice = 29;
        matlabbatch{4}.spm.temporal.st.prefix = 'a';
        %coregistration
        matlabbatch{5}.spm.spatial.coreg.estwrite.ref(1) = cfg_dep('Realign: Estimate & Reslice: Mean Image', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','rmean'));
        matlabbatch{5}.spm.spatial.coreg.estwrite.source = {anatName};
        matlabbatch{5}.spm.spatial.coreg.estwrite.other = {''};
        matlabbatch{5}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
        matlabbatch{5}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
        matlabbatch{5}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
        matlabbatch{5}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
        matlabbatch{5}.spm.spatial.coreg.estwrite.roptions.interp = 4;
        matlabbatch{5}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
        matlabbatch{5}.spm.spatial.coreg.estwrite.roptions.mask = 0;
        matlabbatch{5}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';
        %segment
        matlabbatch{6}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Coregister: Estimate & Reslice: Coregistered Images', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
        matlabbatch{6}.spm.spatial.preproc.channel.biasreg = 0.001;
        matlabbatch{6}.spm.spatial.preproc.channel.biasfwhm = 60;
        matlabbatch{6}.spm.spatial.preproc.channel.write = [0 1];
        matlabbatch{6}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/neurosoft/matlab_tools/spm12/tpm/TPM.nii,1'};
        matlabbatch{6}.spm.spatial.preproc.tissue(1).ngaus = 1;
        matlabbatch{6}.spm.spatial.preproc.tissue(1).native = [1 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(1).warped = [0 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/neurosoft/matlab_tools/spm12/tpm/TPM.nii,2'};
        matlabbatch{6}.spm.spatial.preproc.tissue(2).ngaus = 1;
        matlabbatch{6}.spm.spatial.preproc.tissue(2).native = [1 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(2).warped = [0 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/neurosoft/matlab_tools/spm12/tpm/TPM.nii,3'};
        matlabbatch{6}.spm.spatial.preproc.tissue(3).ngaus = 2;
        matlabbatch{6}.spm.spatial.preproc.tissue(3).native = [1 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(3).warped = [0 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/neurosoft/matlab_tools/spm12/tpm/TPM.nii,4'};
        matlabbatch{6}.spm.spatial.preproc.tissue(4).ngaus = 3;
        matlabbatch{6}.spm.spatial.preproc.tissue(4).native = [1 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(4).warped = [0 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(5).tpm = {'/usr/local/neurosoft/matlab_tools/spm12/tpm/TPM.nii,5'};
        matlabbatch{6}.spm.spatial.preproc.tissue(5).ngaus = 4;
        matlabbatch{6}.spm.spatial.preproc.tissue(5).native = [1 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(5).warped = [0 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(6).tpm = {'/usr/local/neurosoft/matlab_tools/spm12/tpm/TPM.nii,6'};
        matlabbatch{6}.spm.spatial.preproc.tissue(6).ngaus = 2;
        matlabbatch{6}.spm.spatial.preproc.tissue(6).native = [0 0];
        matlabbatch{6}.spm.spatial.preproc.tissue(6).warped = [0 0];
        matlabbatch{6}.spm.spatial.preproc.warp.mrf = 1;
        matlabbatch{6}.spm.spatial.preproc.warp.cleanup = 1;
        matlabbatch{6}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
        matlabbatch{6}.spm.spatial.preproc.warp.affreg = 'eastern';
        matlabbatch{6}.spm.spatial.preproc.warp.fwhm = 0;
        matlabbatch{6}.spm.spatial.preproc.warp.samp = 3;
        matlabbatch{6}.spm.spatial.preproc.warp.write = [0 1];
        %Normalise EPI data using forward deformations
        matlabbatch{7}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{6}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
        matlabbatch{7}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 1)', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
        matlabbatch{7}.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72
                                                                  90 90 108];
        matlabbatch{7}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
        matlabbatch{7}.spm.spatial.normalise.write.woptions.interp = 4;
        matlabbatch{7}.spm.spatial.normalise.write.woptions.prefix = 'w';
        %normalise T1 data
        matlabbatch{8}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{6}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
        matlabbatch{8}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.','val', '{}',{6}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','channel', '()',{1}, '.','biascorr', '()',{':'}));
        matlabbatch{8}.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72
                                                                  90 90 108];
        matlabbatch{8}.spm.spatial.normalise.write.woptions.vox = [1 1 1.33];
        matlabbatch{8}.spm.spatial.normalise.write.woptions.interp = 4;
        matlabbatch{8}.spm.spatial.normalise.write.woptions.prefix = 'w';
        %smooth EPI data
        matlabbatch{9}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{7}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
        matlabbatch{9}.spm.spatial.smooth.fwhm = [6 6 6];
        matlabbatch{9}.spm.spatial.smooth.dtype = 0;
        matlabbatch{9}.spm.spatial.smooth.im = 0;
        matlabbatch{9}.spm.spatial.smooth.prefix = 's6';
        %mooth PEI data2
        matlabbatch{10}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{7}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
        matlabbatch{10}.spm.spatial.smooth.fwhm = [8 8 8];
        matlabbatch{10}.spm.spatial.smooth.dtype = 0;
        matlabbatch{10}.spm.spatial.smooth.im = 0;
        matlabbatch{10}.spm.spatial.smooth.prefix = 's8';

        spm_jobman('run', matlabbatch);
        clear matlabbatch;

        %delete useless files
        delete([funcPath,'rfunc.nii'])
        delete([funcPath,'arfunc.nii'])
        delete([funcPath,'meanfunc.nii'])
        %delete([funcPath,'wrafuc.nii'])
        %delete([funcPath,'swrafunc.nii'])
    else
        continue
    end
    
end
