%% Get posterior pdf summary

function [W, Z, selVars, varargout] = getSummary(results, zThreshold, ploton)
    if(nargin<2)
        zThreshold = 0.5;
        ploton = false;
    end
    if(nargin<3)
        ploton = false;
    end
    
    isig2 = false;
    
    if(~iscell(results))
        Z   = results.samples.z;
        W   = results.samples.w;
        if(isfield(results.samples,'beta'))
            isig2 = results.samples.beta;
        end

    elseif(length(results)==1)
        Z   = results{1}.samples.z;
        W   = results{1}.samples.w;
        if(isfield(results{1}.samples,'beta'))
            isig2 = results{1}.samples.beta;
        end
    else
        nChains   = length(results);
        [nsamp,d] = size(results{1,1}.samples.z);
        Z = nan(nChains,d);
        W = nan(nChains*nsamp,d);
        Wstore = zeros(nsamp, d, nChains);
        if(isfield(results{1,1}.samples,'beta'))
            isig2 = nan(nChains*nsamp,1);
        end    
        for i = 1:nChains
           Z(i,:) = mean(results{i}.samples.z);
           arange = ((i-1)*nsamp + 1) : i*nsamp;
           W(arange,:) = results{i}.samples.w;
           Wstore(:,:,i) = results{i}.samples.w;
           if(isfield(results{1,1}.samples,'beta'))
               isig2(arange,:)= results{i}.samples.beta;
           end
        end
        
        % Multivariate PSRF (gelman-rubin stat)
        Rw = mpsrf_jitter(Wstore);

    end
    
    % Select variables based on Median probability model
    z = mean(Z);
    selVars = find(z > zThreshold);
    
    d = length(z);
    
    if(ploton)
        figure; clf;
        stem(1:d, z,'filled'); hold on
        plot([0, d+1],zThreshold*[1,1], 'r--'); 
        title('Marginal post prob of z')
        grid on;
        xlim([0,d+1]);
        ylim([0,1.1]);
        set(gca, 'Fontsize',12)
    end
    
    varargout{1} = isig2;
    
    if(length(results)~=1)
        varargout{2} = Rw;
    end
%     figure; clf;
%     stem(1:34, mean(ZZ),'filled'); hold on
%     plot([0, 35],0.5*[1,1], 'g--'); 
%     ylabel('$p(z_i=1 \mid y)$','interpreter','latex'); xlabel('Variable \it i')
%     grid on;
%     xlim([0,34+1]);
%     ylim([0,1.1]);
%     set(gca, 'Fontsize',12)
end   
