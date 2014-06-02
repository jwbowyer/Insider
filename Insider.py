# -*- coding: iso-8859-15 -*-
#version 0.1

#import Xrawler
import pickle as pkl
import os
import sys
import math
import numpy as np
from lxml.html import parse
from urllib2 import urlopen
import matplotlib.pyplot as plt
import scipy.stats as stat
import time

import warnings

def fix_div_by_zero(array):

 new_array = []
 #new_array = replace_inf_with_nan(array)	#some numpy code

 return new_array

def fxn():
 warnings.warn("deprecated", DeprecationWarning)

def hist_subs(data,binStart,binEnd,nbins,nrows=1,ncols=1,pltorder=1):

 increment = float(binEnd-binStart)/nbins
 bins=[ binStart+increment*i for i in range(nbins+1)]

 facec=['blue','green','red','purple']
 plt.subplot(nrows,ncols,pltorder)
 plt.hist(data,bins=bins,range=[binStart,binEnd],facecolor=facec[pltorder-1], alpha=0.5)

def get_fit(data,binStart,binEnd,nbins,fit_type='norm'):

 ff=[]
 try:
  fg = getattr(stat, fit_type)
  ff = fg.fit(data)
 except:
  print "Invalid fit_type choice" 
  return 0

 fit_info = [ f for f in ff]	#tuple to list

 increment = float(binEnd-binStart)/nbins
 bins=[ binStart+increment*i for i in range(nbins+1)]

 #Make this bit prettier/generalized
 pdf=0
 try:
  fg = getattr(stat, fit_type)
  if len(fit_info)==4:
   pdf = fg.pdf(bins,fit_info[0],fit_info[1],fit_info[2],fit_info[3])
  elif len(fit_info)==3:
   pdf = fg.pdf(bins,fit_info[0],fit_info[1],fit_info[2])
  elif len(fit_info)==2:
   pdf = fg.pdf(bins,fit_info[0],fit_info[1])
  elif len(fit_info)==1:
   pdf = fg.pdf(bins,fit_info[0])
 except:
  print "Invalid fit_type choice" 
  return 0

 return bins,pdf

def fit_plot(data,binStart,binEnd,nbins,fit_type='norm'):

 #docs.scipy.org/doc/scipy-0.13.0/reference/stats.html
 #en.wikipedia.org/wiki/List_of_probability_distributions
 data_filt = [ dat for dat in data if dat>=binStart and dat<=binEnd ]
 hist_subs(data_filt,binStart,binEnd,nbins,1,1,1)

 return get_fit(data_filt,binStart,binEnd,nbins,fit_type)

def arxiv_to_date(papers):

 dates = np.array([ (paper.split(".")[0]) for paper in papers ])
 order = np.array([ (paper.split(".")[1]) for paper in papers ])

 dconv=[]
 for i in range(len(dates)):
  dconv.append([j for j in dates[i]])
  dconv[i]=float("".join(dconv[i][:2]))+float("".join(dconv[i][2:]))/12.0
  dconv[i]=dconv[i]+(1.0/12.0)*(float(order[i])/10000.0)

 return dconv


def flatten(x):
    #from stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def pkldump(ffile,array):

 with open(ffile,"wb") as ff:
  pkl.dump(array,ff,pkl.HIGHEST_PROTOCOL)

def pklload(ffile):

 array = []
 with open(ffile,"r") as ff:
  array = pkl.load(ff)
 return array

def get_authors(paper):

 author_list = []
 url="http://arxiv.org/abs/"+paper
 tables = []

 time.sleep(np.random.random_integers(1,10))
 try:
  parsed = parse(urlopen(url))
  doc = parsed.getroot()
  tables = doc.findall('.//meta')
 except:
  print "cannot get author list"

 j=1					#Authors always start at 1
 while tables[j].get("name")=="citation_author":
  author_list.append(tables[j].get("content"))
  j+=1

 return author_list 

def get_papers(ffile,url):

 papersCleaned = []
 fromFile=0
 if os.path.isfile(ffile)==1:
   if force_get==1:
    os.system("rm "+ffile)
   else:
    fromFile=1

 time.sleep(np.random.random_integers(1,10))
 if fromFile==0:

  parsed = parse(urlopen(url))
  doc = parsed.getroot()
  tables = doc.findall('.//dt')
  nPapers=len(tables)

  #Get and clean papers data
  papers = [ tables[i].text_content() for i in range(nPapers) ]
  papersCleaned = [ (p.split()[1]).replace("arXiv:","") for p in papers ]

  pkldump(ffile,papersCleaned)

 else:

  papersCleaned = pklload(ffile)

 return papersCleaned

def impact(words=["BICEP2","B-modes"],collab=[],force_get=0,exptDate="1000.1000",pagelimit=250,plots=1):
 #Impact and insiders

 dir_spec = [ i.replace(" ","_") for i in words ]
 words=dir_spec
 dir_spec = [ i.replace("/","_") for i in words ]
 words=dir_spec
 dir_spec=""
 for i in words[:-1]:
  dir_spec += i+"_"
 dir_spec += words[-1]

 data_dir=dir_spec+"_data/"
 d = os.path.dirname(data_dir)
 if not os.path.exists(d):
  os.makedirs(d)

 srchTypes=['all','ti','abs','au']		#search all,titles,abstracts,authors 
 #arxiv.org/find/all/1/ti:+bicep2/0/1/0/all/0/1	#ti:(BICEP2 AND b-modes)
 #arxiv.org/find/all/1/AND+ti:+BICEP2+abs:+b-modes/0/1/0/all/0/1
 #search.arxiv.org:8081/?query=bicep2&in=		#fulltext search
 #arxiv.org/find/all/1/abs:+bicep2/0/1/0/all/0/1?per_page=100

 #Generalize this
 url="http://arxiv.org/find/all/1/"+srchTypes[0]+":+"+words[0].lower()+"/0/1/0/all/0/1?per_page="+str(pagelimit)
 ffile=data_dir+dir_spec+"papers_data.pkl"
 papersCleaned = get_papers(ffile,url)

 #From this alone, we can plot a histogram of activity.. compare with B-modes/polarization/cosmo activity overall.

 #Although there is no fine-grained "day published" data, I can assume that they get published in numerical order.
 #So, a fudge is to do order*30/10000.. although there are never the same number published per day, and the limit is not reached.
 dconv=arxiv_to_date(papersCleaned)

 exptDate = "1403.3985".split(".")[0]
 exptDate = [j for j in exptDate]
 exptDate = float("".join(exptDate[:2]))+float("".join(exptDate[2:]))/12.0
 exptDate = exptDate + (1.0/12.0)*(float("1403.3985".split(".")[1])/10000.0)

 hist_subs(dconv,10,15,40,2,2,1)		#I want to automate finding the spike point when responses start
 plt.ylabel('No. preprints')

 hist_subs(dconv,14,14.5,40,2,2,2)
 #So we can see the flurry of activity post-BICEP2 b-modes pubn

 #Now plot before/after
 print "Data release at "+str(exptDate)
 #exptDate = math.floor(exptDate*4)/4	#approximaate it
 hist_subs(dconv,10,exptDate,40,2,2,3)
 plt.ylabel('No. preprints')
 plt.xlabel('Date')
 hist_subs(dconv,exptDate,14.5,40,2,2,4)

 plt.suptitle(dir_spec+' publication activity') 

 plt.xlabel('Date')
 #if plots==1:
 # plt.show()

 #Now let's fit a distribution to the histogram; this will allow us to define the short-vs-long term impact
 binStart=14.0
 binEnd=14.5
 nbins=40
 #Just strip this from the web.. I actually want a distribution that spikes first.
 fits_cont = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','cauchy','chi','chi2','cosine','dgamma','dweibull','erlang',
  'expon','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','genpareto',
  'genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy',
  'halflogistic','halfnorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','ksone','kstwobign','laplace','logistic',
  'loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw',
  'powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm',
  'tukeylambda','uniform','vonmises','wald','weibull_min','weibull_max','wrapcauchy']
 fits_discr = ['bernoulli','binom','boltzmann','dlaplace','geom','hypergeom','logser','nbinom','planck','poisson','randint','skellam','zipf']

 #Apply tests to this fit --- how good is it? Start with goodness-of-fit (chi-square), or use R
 #Note that scipy doesn't contain Cramer-von Mises test
 pmax=0
 fit_best=""
 with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  fxn()
  for fit in fits_cont:

   #p-value <0.05, reject the null hypothesis.. check that I know what I'm doing. 2nd value is always p
   try:
    bins, pdf = get_fit(dconv,binStart,binEnd,40,fit)
    n,p1 = stat.pearsonr(bins,pdf)
    if p1<0.05:		#fails pearson test
     continue    
    n,p2 = stat.chisquare(bins,pdf)
    if p1<0.05:		#fails chi-square test
     continue    
    n,p3 = stat.spearmanr(bins,pdf)
    if p1<0.05:		#fails spearman test
     continue    

    #use rms to choose fit
    rms = sqrt(p1**2 + p2**2 +p3**2)
    if rms>pmax:
     pmax=rms
     fit_best=fit

   except:
    print "Cannot use "+fit
    pass

 if fit_best=="":
  print "No good fits (pmax = "+str(pmax)+") --- adjust your data, fit types, or acceptance criteria."
  #sys.exit()
 print "Best fit by "+fit_best

 #Instead, think about the underlying "physics"
 bins, pdf = fit_plot(dconv,binStart,binEnd,40,fit_best)
 #bins, pdf = fit_plot(dconv,binStart,binEnd,40,'norm')

 plt.plot(bins, pdf, 'r--', linewidth=2)
 if plots==1:
  plt.show()

 sys.exit()
 
 #Now we run through each paper, getting the authors and citations.

 fullAuthors=[]
 nCites=[]
 nPapers=len(papersCleaned)

 fromFile=0
 ffile=data_dir+dir_spec+"authors_data.pkl"
 ffile2=data_dir+dir_spec+"cites_data.pkl"
 if os.path.isfile(ffile)==1:
  if force_get==1:
   os.system("rm "+ffile)
   os.system("rm "+ffile2)
  else:
   fromFile=1

 if fromFile==0:

  print "Iterating over "+str(nPapers)+" papers"
  for i in range(nPapers):
	print i
	fullAuthors.append(get_authors[papersCleaned[i]])

  pkldump(ffile,fullAuthors)

  for i in range(nPapers):
	time.sleep(np.random.random_integers(1,10))
	#Citations: not very reliable.
	url="http://arxiv.org/cits/"+papersCleaned[i]
	parsed = parse(urlopen(url))
	doc = parsed.getroot()
	tables = doc.findall('.//dt')
	nCites.append(len(tables))

  pkldump(ffile2,nCites)

 else:

  fullAuthors = pklload(ffile)
  nCites = pklload(ffile2)

 #now flatten the authors list, and get the unique elements
 fullAuthors = flatten(fullAuthors)
 fullAuthors = list(set(fullAuthors))
 #Check that fullAuthors does not contain any collab members.

 nAuthors=len(fullAuthors)
 nCollab=len(collab)

 #Get the collaboration frequencies between the remaining authors and the collab members. Might require time-correlation too.
 BRfreqs=np.zeros([nAuthors,nCollab])

 #Get limited authors list in URL-processable form --- make more compact!
 #ISSUES: Does author go by 1st or 2nd name? How do I deal with duplicate names in the research community?
 for i in range(nCollab):
  collab[i] = ( (collab[i].split())[0].strip(",").lower().replace("'",""), (collab[i].split())[1].strip(".").lower() )

 i=0
 while i<nAuthors:
  fullAuthors[i] = [ (fullAuthors[i].split())[0].strip(",").lower(), fullAuthors[i].split()[1][0].lower() ]
  if fullAuthors[i][0]==words[0].lower():
   fullAuthors.pop(i)
   nAuthors -= 1
  i += 1

 for i in range(nAuthors):
  fullAuthors[i] = " ".join(fullAuthors[i])
 fullAuthors = list(set(fullAuthors))	#Ensure no unnecessary removals. Remember that sset is unordered.

 nAuthors=len(fullAuthors)
 for i in range(nAuthors):
  fullAuthors[i] = ( (fullAuthors[i].split())[0], (fullAuthors[i].split())[1] )

 for i in range(nCollab):
  try:
   fullAuthors.remove(collab[i])
  except:
   pass

 nAuthors=len(fullAuthors)
 #Need to do ş->s, ğ->g, etc; might want to do this earlier, for removing doubles
 from unidecode import unidecode
 for i in range(nAuthors):
  fullAuthors[i]=(unidecode(fullAuthors[i][0]),unidecode(fullAuthors[i][1]))	#runtime warning.. check!

 fromFile=0
 ffile=data_dir+"BR_corrn.pkl"
 if os.path.isfile(ffile)==1:
  if force_get==1:
   os.system("rm "+ffile)
  else:
   fromFile=1

 sys.exit()

 #uk.arxiv.org/find/all/1/au:+AND+lange_a+yoon_k/0/1/0/all/0/1?per_page=250
 #I don't necessarily need the sparse array to find the insider effect: just the sum over collaborations (unless I want the source).
 if fromFile==0:

  print "Iterating over "+str(nAuthors*nCollab)+" author combinations"
  #Blocked @1771: don't need to search by such a fine comb.
  """for i in range(nAuthors):
   for j in range(nCollab):
    url="http://arxiv.org/find/all/1/"+srchTypes[3]+":+AND+"+fullAuthors[i][0]+"_"+fullAuthors[i][1]+"+"+collab[j][0]+"_"+collab[j][1]+"/0/1/0/all/0/1?per_page="+str(pagelimit)

    print url
    parsed = parse(urlopen(url))
    doc = parsed.getroot()
    tables = doc.findall('.//h3')
    if tables[0].text_content()!="Search gave no matches":
     tables = doc.findall('.//dt')
     BRfreqs[i][j]=len(tables)
    print str(i*nCollab+j)+": "+str(BRfreqs[i][j])"""

  #Example calls:
  #(jaffe AND novikov):					arxiv.org/find/all/1/au:+AND+jaffe+novikov/0/1/0/all/0/1
  #(jaffe AND (novikov OR chamballu)):			arxiv.org/find/all/1/au:+AND+jaffe+OR+novikov+chamballu/0/1/0/all/0/1
  #(jaffe AND novikov) OR chamballu:			arxiv.org/find/all/1/au:+OR+AND+jaffe+novikov+chamballu/0/1/0/all/0/1
  #(jaffe AND ((novikov OR chamballu) OR bowyer):	arxiv.org/find/all/1/au:+AND+jaffe+OR+OR+novikov+chamballu+bowyer/0/1/0/all/0/1
  #(jaffe AND (novikov OR (chamballu OR bowyer)):	arxiv.org/find/all/1/au:+AND+jaffe+OR+novikov+OR+chamballu+bowyer/0/1/0/all/0/1
  #(jaffe AND novikov) OR (jaffe AND chamballu):	arxiv.org/find/all/1/au:+OR+AND+jaffe+novikov+AND+jaffe+chamballu/0/1/0/all/0/1

  BRfreqs=np.zeros([nAuthors])

  for i in range(nAuthors):
    allCollab="+"
    for j in range(nCollab-1):
     allCollab += "OR+AND+"+fullAuthors[i][0]+"_"+fullAuthors[i][1]+"+"+collab[j][0]+"_"+collab[j][1]+"+"
    allCollab += "AND+"+fullAuthors[i][0]+"_"+fullAuthors[i][1]+"+"+collab[j+1][0]+"_"+collab[j+1][1]

    url="http://arxiv.org/find/all/1/"+srchTypes[3]+":"+allCollab+"/0/1/0/all/0/1?per_page="+str(pagelimit)
    time.sleep(np.random.random_integers(1,10))
    #print url
    try:
     parsed = parse(urlopen(url))
     doc = parsed.getroot()
     tables = doc.findall('.//h3')

     if tables[0].text_content()!="Search gave no matches":
      tables = doc.findall('.//dt')
      BRfreqs[i]=len(tables)
     print str(i)+": "+str(BRfreqs[i])
    except:
     print str(i)+": Could not grab for "+fullAuthors[i][0]+"_"+fullAuthors[i][1]
     pass

  pkldump(ffile,BRfreqs)	#add force_recalc params.. need to be able to write in data that I couldn't grab previously

 else:

  BRfreqs = pklload(ffile)

 #What is the correlation between n_collab_authorship and time-from-source-publication?
 #We can then compare across experiments
 #Also, those collaborations that are far n-sigma out can be considered insiders
 

 #Second approach: Are the BR frequencies correlated with the timing/number of BICEP postings by that outsider?
 #Plot the time-distribution of the top authors' postings, and overlap with BICEP. I could restrict to BICEP-related papers [CHECK]

 threshold=0.1
 j=1
 authors_check=[]
 for i in range(nAuthors):		#do from most to least authorship
  url="http://arxiv.org/find/all/1/"+srchTypes[3]+":+"+fullAuthors[i][0]+"_"+fullAuthors[i][1]+"/0/1/0/all/0/1?per_page="+str(pagelimit)
  time.sleep(np.random.random_integers(1,10))
  try:
   parsed = parse(urlopen(url))
   doc = parsed.getroot()
   tables = doc.findall('.//dt')
   nAuthPapers=len(tables)

   #Get and clean papers data
   authPapers = [ tables[i].text_content() for i in range(nAuthPapers) ]
   authPapersCleaned = [ (p.split()[1]).replace("arXiv:","") for p in papers ]

   dconvA=arxiv_to_date(authPapersCleaned)

   hist_subs(dconvA,10,15,40,nAuthors,2,j)
   j+=1
   plt.ylabel('Preprints for author '+str(i))

   #code to compare distributions. Assuming we have the right fit, we look for a pdf1/pdf2 at the time of BICEP publication
   #Will need to coarse-grain; I then list the absmax of 1-pdf1/pdf2 and choose the smallest values. Possibly shift the datet to BICEPdate?

   data_filt = [ dat for dat in dconvA if dat>=binStart and dat<=binEnd ]
   bins2, pdf2 = get_fit(data_filt,14.0,14.5,40,fit_best)
   pdf_diff = abs(pdf/pdf2)
   #pdf_diff = fix_div_by_zero(abs(pdf/pdf2))

   plt.plot(bins, pdf_diff, 'r--', linewidth=2)
   authors_check.append(min(1-pdf_diff))

   #If suspicious, we can look at the (1) time spread of author pubns, and that of collab with BR.
   if authors_check[i] < threshold:

  except:
   print str(i)+": Could not grab for "+fullAuthors[i][0]+"_"+fullAuthors[i][1]
   pass


 plt.xlabel('Date') 
 plt.xlabel('Correlation with BICEP') 
 if plots==1:
  plt.show()

 #Can also get the intercollabs between the remaining authors (second-order insiders)
 autocorr=0
 if autocorr==1:
  fromFile=0
  RRfreqs=np.zeros([nAuthors,nAuthors])
  ffile=data_dir+"RR_corrn.pkl"
  if os.path.isfile(ffile)==1:
   if force_get==1:
    os.system("rm "+ffile)
   else:
    fromFile=1

  if fromFile==0:

   print "Iterating over "+str(nAuthors**2)+" author combinations" 
   for i in range(nAuthors):
    for j in range(i):
     print i*nAuthors+j
     url="http://arxiv.org/find/all/1/"+srchTypes[3]+":+AND+"+fullAuthors[i]+"+"+fullAuthors[j]+"/0/1/0/all/0/1?per_page="+str(pagelimit)
     parsed = parse(urlopen(url))
     doc = parsed.getroot()
     tables = doc.findall('.//h3')
     if tables[0].text_content()!="Search gave no matches":
      tables = doc.findall('.//dt')
      RRfreqs[i][j]=len(tables)
     print str(i*nAuthors+j)+": "+str(RRfreqs[i][j])

   pkldump(ffile,RRfreqs)

  else:

   RRfreqs = pklload(ffile)

 #Our arrays are sparse, so it's better to just have the index and the value (for non-zero elements)

 #We should discriminate between papers in direct response to BICEP, and those that only include/use BICEP by-the-by. Maybe by count of BICEP mentions? (Particularly the title)

 #Plot vs time to see whether there are "insiders"!

 #Get the author pairs that these correspond to. Check their distance from BICEP.

#Compare with previous exciting/non-exciting CMB expts to get a sense of normalcy, i.e. BICEP1
BICEPdate="1403.3985"

#List BICEP members and exclude them and their papers from the lists.
#BICEP=get_authors(BICEPdate)	#any problems with 'Ogburn IV, R. W.'?
BICEP=[]

words=["BICEP2"]
impact(words,BICEP,0,BICEPdate,250,1)


"""takahashi_y+ade_p -> 9.0
takahashi_y+aikin_r -> 2.0
takahashi_y+barkats_d -> 9.0
takahashi_y+bock_j -> 9.0
takahashi_y+buder_i -> 2.0
takahashi_y+dowell_c -> 9.0
takahashi_y+duband_l -> 9.0
takahashi_y+filippini_j -> 2.0
takahashi_y+hristov_v -> 9.0
takahashi_y+keating_b -> 9.0
takahashi_y+kernasovskiy_s -> 2.0
takahashi_y+kovac_j -> 9.0
takahashi_y+kuo_c -> 9.0
takahashi_y+leitch_e -> 9.0
takahashi_y+mason_p -> 9.0
takahashi_y+nguyen_h -> 9.0
takahashi_y+pryke_c -> 7.0
takahashi_y+richter_s -> 7.0
takahashi_y+sheehy_c -> 5.0
takahashi_y+tolan_j -> 4.0
takahashi_y+yoon_k -> 9.0
lee_s+nguyen_h -> 1.0
lee_s+richter_s -> 3.0
lee_s+wong_c -> 3.0
anchordoqui_l+fliescher_s -> 24.0
lee_w+nguyen_h -> 1.0
ferreira_p+ade_p -> 21.0
ferreira_p+bock_j -> 20.0
ferreira_p+buder_i -> 1.0
ferreira_p+halpern_m -> 1.0
ferreira_p+hilton_g -> 1.0
ferreira_p+hristov_v -> 18.0
ferreira_p+irwin_k -> 1.0
ferreira_p+mason_p -> 4.0
ferreira_p+netterfield_c -> 14.0
ferreira_p+nguyen_h -> 1.0
ferreira_p+reintsema_c -> 1.0
ferreira_p+sudiwala_r -> 1.0
ferreira_r+nguyen_h -> 1.0
lee_h+ade_p -> 1.0
lee_h+bock_j -> 2.0
lee_h+dowell_c -> 1.0
lee_h+halpern_m -> 2.0
lee_h+netterfield_c -> 1.0
lee_h+nguyen_h -> 2.0
lee_h+richter_s -> 1.0
moss_a+ade_p -> 14.0
moss_a+bock_j -> 19.0
moss_a+hildebrandt_s -> 20.0
moss_a+netterfield_c -> 18.0
moss_a+sudiwala_r -> 13.0
kinney_w+bock_j -> 2.0
kinney_w+buder_i -> 1.0
kinney_w+golwala_s -> 1.0
kinney_w+halpern_m -> 1.0
kinney_w+hildebrandt_s -> 1.0
kinney_w+irwin_k -> 2.0
kinney_w+keating_b -> 2.0
kinney_w+kovac_j -> 2.0
kinney_w+kuo_c -> 1.0
kinney_w+leitch_e -> 2.0
kinney_w+nguyen_h -> 2.0
kinney_w+orlando_a -> 1.0
kinney_w+pryke_c -> 2.0
kinney_w+vieregg_a -> 1.0
kinney_w+yoon_k -> 1.0
gao_x+ade_p -> 1.0
gao_x+halpern_m -> 1.0
gao_x+hilton_g -> 1.0
gao_x+irwin_k -> 1.0
gao_x+reintsema_c -> 1.0
gao_x+sudiwala_r -> 1.0
zurek_k+golwala_s -> 1.0
zaldarriaga_m+ade_p -> 4.0
zaldarriaga_m+barkats_d -> 1.0
zaldarriaga_m+bock_j -> 3.0
zaldarriaga_m+buder_i -> 1.0
zaldarriaga_m+golwala_s -> 1.0
zaldarriaga_m+halpern_m -> 1.0
zaldarriaga_m+hildebrandt_s -> 1.0
zaldarriaga_m+hilton_g -> 1.0
zaldarriaga_m+irwin_k -> 3.0
zaldarriaga_m+keating_b -> 2.0
zaldarriaga_m+kovac_j -> 2.0
zaldarriaga_m+kuo_c -> 1.0
zaldarriaga_m+leitch_e -> 2.0
zaldarriaga_m+nguyen_h -> 2.0
zaldarriaga_m+orlando_a -> 1.0
zaldarriaga_m+pryke_c -> 2.0
zaldarriaga_m+reintsema_c -> 1.0
zaldarriaga_m+vieregg_a -> 1.0
zaldarriaga_m+yoon_k -> 1.0
lange_a+ade_p -> 91.0
lange_a+aikin_r -> 2.0
lange_a+barkats_d -> 6.0
lange_a+benton_s -> 3.0
lange_a+bock_j -> 97.0
lange_a+dowell_c -> 8.0
lange_a+duband_l -> 8.0
lange_a+filippini_j -> 2.0
lange_a+golwala_s -> 12.0
lange_a+halpern_m -> 4.0
lange_a+hasselfield_m -> 2.0
lange_a+hildebrandt_s -> 3.0
lange_a+hilton_g -> 4.0
lange_a+hristov_v -> 49.0
lange_a+irwin_k -> 6.0
lange_a+keating_b -> 16.0
lange_a+kovac_j -> 20.0
lange_a+kuo_c -> 21.0
lange_a+leitch_e -> 19.0
lange_a+lueker_m -> 8.0
lange_a+mason_p -> 23.0
lange_a+netterfield_c -> 43.0
lange_a+nguyen_h -> 15.0
lange_a+ogburn_r -> 3.0
lange_a+orlando_a -> 14.0
lange_a+pryke_c -> 17.0
lange_a+reintsema_c -> 2.0
lange_a+richter_s -> 5.0
lange_a+schwarz_r -> 11.0
lange_a+sheehy_c -> 3.0
lange_a+staniszewski_z -> 3.0
lange_a+sudiwala_r -> 12.0
lange_a+teply_g -> 2.0
lange_a+tolan_j -> 2.0
lange_a+turner_a -> 18.0
lange_a+wong_c -> 1.0
lange_a+yoon_k -> 6.0
"""
