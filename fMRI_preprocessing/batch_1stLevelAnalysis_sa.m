clear;
clc;

[num,txt,raw] = xlsread('/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/fileDetection.xls');
subjs = raw(2:end,1);%242 failed, 231x excluded 
selection = raw(2:end,4);%head movetion
behavior = raw(2:end,5);%behavior performance
design = raw(2:end,6);

%design for central,R_Peripheral,L_peripheral
design1 = {[20-8 38-8 110-8 146-8 164-8 200-8 272-8 290-8],[56-8 92-8 218-8 254-8 74-8 128-8 182-8 236-8]};
design2 = {[20-8 56-8 128-8 146-8 164-8 182-8 254-8 290-8],[74-8 110-8 200-8 236-8 38-8 92-8 218-8 272-8]};

error = {};    
for i = 1:length(subjs)
    subj = subjs{i};
    if selection{i} == 1 && behavior{i} == 1
        try
            %input fMRI data
            for j = 1:158
                funcName{j} =  ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/data/',subj,'/sa/s8warfunc.nii,',num2str(j)];
            end

            savePath = ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/data/',subj,'/1st_sa'];
            if ~exist(savePath)
                mkdir(savePath);
            end

            regPath = ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/data/',subj,'/sa/rp_func24.txt'];

            matlabbatch{1}.spm.stats.fmri_spec.dir = {savePath};
            matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
            matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 30;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 29;
            %%
            matlabbatch{1}.spm.stats.fmri_spec.sess.scans = funcName';
            %%
            if design{i} == 1
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'central';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = design1{1};
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = 16;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'peripheral';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = design1{2};
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = 16;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
                matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regPath};
                matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
                matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
                matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
                matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
                matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
                matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
                matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
                matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
            elseif design{i} == 2
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'central';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = design2{1};
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = 16;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'peripheral';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = design2{2};
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = 16;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
                matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regPath};
                matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
                matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
                matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
                matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
                matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
                matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
                matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
                matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
            end    
            
                
            %model estimation
            matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;      


            spm_jobman('run', matlabbatch);
            clear matlabbatch;
        catch
            error = [error,subj];
        end
    else
        continue
    end
end
