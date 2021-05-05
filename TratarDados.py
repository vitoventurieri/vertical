from os import walk
from os.path import splitext
from os.path import join
import glob
import pandas as pd
foodir = 'output'
City = list()
Ward = list()
ICU = list()
WardPeakNoIsolation = list()
ICUPeakNoIsolation = list()
WardPeakVertical = list()
ICUPeakVertical = list()
Deceased = list()
R = list()
HpeakNoIsolation = list()
UpeakNoIsolation = list()
HpeakVertical = list()
UpeakVertical = list()
#FileNames = ['results_medians_no_isolation__confidence_interval','results_percentile_05_no_isolation__confidence_interval',
#'results_percentile_95_no_isolation_confidence_interval','results_medians_vertical__confidence_interval',
#'results_percentile_05_vertical__confidence_interval','results_percentile_95_vertical__confidence_interval']

for root, dirs, files in walk(foodir):
#    for d in dirs:
#        df = pd.read_excel(join(root,d,'parameters__confidence_interval.xlsx'))
#        WardBeds = df.iat[22,1]
#        ICUBeds = df.iat[23,1]

#        resultsH = list()
#        resultsU = list()
#        for result in FileNames:
#            df = pd.read_excel(join(root,d,result+'.xlsx'))
#            H = df['Hi']+df['Hj']
#            U = df['Ui']+df['Uj']            
#            Hpeak = max(H)
#            Upeak = max(U)
#            resultsH.append(Hpeak/WardBeds)
#            resultsU.append(Upeak/ICUBeds)
#        stringHNoIsolation = str(round(resultsH[0],2))+ ' (' + str(round(resultsH[1],2)) + ' - ' + str(round(resultsH[2],2)) + ')'
#        stringUNoIsolation = str(round(resultsU[0],2))+ ' (' + str(round(resultsU[1],2)) + ' - ' + str(round(resultsU[2],2)) + ')'
#        WardPeakNoIsolation.append(stringHNoIsolation)
#        ICUPeakNoIsolation.append(stringUNoIsolation)
#        stringHVertical = str(round(resultsH[3],2))+ ' (' + str(round(resultsH[4],2)) + ' - ' + str(round(resultsH[5],2)) + ')'
#        stringUVertical = str(round(resultsU[3],2))+ ' (' + str(round(resultsU[4],2)) + ' - ' + str(round(resultsU[5],2)) + ')'
#        WardPeakVertical.append(stringHVertical)
#        ICUPeakVertical.append(stringUVertical)
    for d in dirs:    
        FileNameParameters = glob.glob(root + '/' + d + '/parameters*.xlsx')
        df = pd.read_excel(FileNameParameters[0])
        WardBeds = df.iat[22,1]
        ICUBeds = df.iat[23,1]
        FileNameParameters = glob.glob(root + '/' + d + '/comparison*.txt')
        print(FileNameParameters)

#    for f in files:            
#        if splitext(f)[1].lower() == ".txt":
        fileTXT = open(FileNameParameters[0],'r')
        City.append(d[32:].split('/')[0])
        lines=fileTXT.readlines()
#            RText = float(lines[2][10:].split()[0])+float(lines[3][10:].split()[0])
#            RText = 1 - RText/(float(lines[13][10:].split()[0])+float(lines[14][10:].split()[0]))
#            R.append(round(100*RText,2))
#            WardText = lines[45][22:].split()
#            Ward.append(round(100*float(WardText[0]),2))
#            ICUText = lines[49][21:].split()
#            ICU.append(round(100*float(ICUText[0]),2))
#            print(files)
        DeceasedText = lines[27][7:]
        Deceased.append(round(100*float(DeceasedText),2))

        MedianVerticalH = float(lines[6][7:].split()[0])/WardBeds
        MedianNoIsolationH = float(lines[15][7:].split()[0])/WardBeds

        LowICVerticalH = float(lines[6][7:].split(':')[1].split('-')[0])/WardBeds
        LowICNoIsolationH = float(lines[15][7:].split(':')[1].split('-')[0])/WardBeds

        HighICVerticalH = float(lines[6][7:].split('-')[2].split(')')[0])/WardBeds
        HighICNoIsolationH = float(lines[15][7:].split('-')[2].split(')')[0])/WardBeds

        MedianVerticalU = float(lines[7][14:].split()[0])/ICUBeds
        MedianNoIsolationU = float(lines[16][14:].split()[0])/ICUBeds

        LowICVerticalU = float(lines[7][7:].split(':')[1].split('-')[0])/ICUBeds
        LowICNoIsolationU = float(lines[16][7:].split(':')[1].split('-')[0])/ICUBeds

        HighICVerticalU = float(lines[7][7:].split('-')[2].split(')')[0])/ICUBeds
        HighICNoIsolationU = float(lines[16][7:].split('-')[2].split(')')[0])/ICUBeds
            
        HpeakNoIsolation.append(str(round(MedianNoIsolationH,2)).replace('.',',') + 
            ' (' + str(round(LowICNoIsolationH,2)).replace('.',',') + 
            '-' + str(round(HighICNoIsolationH,2)).replace('.',',') + ')')
        UpeakNoIsolation.append(str(round(MedianNoIsolationU,2)).replace('.',',') + 
            ' (' + str(round(LowICNoIsolationU,2)).replace('.',',') + 
            '-' + str(round(HighICNoIsolationU,2)).replace('.',',') + ')')
        HpeakVertical.append(str(round(MedianVerticalH,2)).replace('.',',') + 
            ' (' + str(round(LowICVerticalH,2)).replace('.',',') + 
            '-' + str(round(HighICVerticalH,2)).replace('.',',') + ')')
        UpeakVertical.append(str(round(MedianVerticalU,2)).replace('.',',') + 
            ' (' + str(round(LowICVerticalU,2)).replace('.',',') + 
            '-' + str(round(HighICVerticalU,2)).replace('.',',') + ')')
        fileTXT.close()
#df = pd.DataFrame(list(zip(R,WardPeak,ICUPeak,Ward,ICU,Deceased)),index = City,
#    columns=['Removidos (%)','Pico leitos','Pico UTI','Leitos (%)','UTI (%)','Óbitos (%)'])
df = pd.DataFrame(list(zip(HpeakNoIsolation,HpeakVertical,UpeakNoIsolation,UpeakVertical,Deceased)),index = City,
    columns=['Sem isolamento','Vertical','Sem isolamento','Vertical','Redução Óbitos (%)'])
df.to_excel('Tabela.xlsx')
print(df)