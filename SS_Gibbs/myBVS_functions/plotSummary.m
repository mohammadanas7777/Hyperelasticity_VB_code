%% Plot weight summary

function W = plotSummary(results)

    if(~iscell(results))
        Z   = results.samples.z;
        W   = results.samples.w;
        
        z = median(Z);
    elseif(length(results)==1)
        Z   = results{1}.samples.z;
        W   = results{1}.samples.w;
        
        z = median(Z);
    else
        nChains   = length(results);
        [nsamp,d] = size(results{1,1}.samples.z);
        Z = nan(nChains,d);
        W = nan(nChains*nsamp,d);
        for i = 1:nChains
           Z(i,:) = median(results{i}.samples.z);
           arange = ((i-1)*nsamp + 1) : i*nsamp;
           W(arange,:) = results{i}.samples.w;  
        end
    
        z = median(Z);
    end
    
    r = find(z>0);
    
    figure(51); clf;
    stem(median(W),'filled','LineStyle','-.','LineWidth',2); 
    title('Median values of w');
    
    
	plotsperFig = 4;
    numfigs = ceil(length(r)/plotsperFig);
    rem = mod(length(r),plotsperFig);
    
    jj = 0;
    for ifig = 1:numfigs
        figure(51+ifig); clf;
        h = [];
        if(ifig == numfigs && (rem ~= 0))
            plotsperFig = rem;
        end
        for iplot = 1:plotsperFig
            jj = jj + 1;
            h(iplot) = subplot(plotsperFig, 1, iplot);
            histogram(W(:,r(jj)),500, 'Normalization','pdf','EdgeColor', 'none');
            ylabel(['w_{', num2str(r(jj)),'}']);
        end
        linkaxes(h,'x');
    end
            
end   
