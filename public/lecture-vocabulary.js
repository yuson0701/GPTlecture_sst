// Locally authored candidate vocabulary. Topic associations are suggestions,
// not words verified in a particular lecturer's slides or audio.
export const vocabularyReferences = [
  ['NHLBI · 심장', 'https://www.nhlbi.nih.gov/health/heart/heart-beats'],
  ['NHLBI · 호흡기', 'https://www.nhlbi.nih.gov/health/lungs'],
  ['NIDDK · 소화기', 'https://www.niddk.nih.gov/health-information/digestive-diseases'],
  ['NCI · 생의학 용어', 'https://www.cancer.gov/publications/dictionaries'],
];
const topics = [
  [/심전도|전기활동|부정맥|빈맥|서맥|심방세동|심박동기/i, '심전도,ECG,EKG,활동전위,action potential,탈분극,재분극,동방결절,SA node,방실결절,AV node,히스속,Purkinje fiber,P wave,QRS complex,QT interval,심방세동,atrial fibrillation'],
  [/심장근|수축조절|심장능력|혈류역학|심부전/i, '심근,cardiomyocyte,수축력,contractility,전부하,preload,후부하,afterload,심박출량,cardiac output,일회박출량,stroke volume,박출률,ejection fraction,Frank-Starling,칼슘,트로포닌,액틴,미오신'],
  [/심장.*구조|심장.*발생|선[천청]성.*심질환/i, '심방,atrium,심실,ventricle,심방중격,심실중격,심내막,endocardium,심근,myocardium,심외막,epicardium,난원공,동맥관,심실중격결손,VSD'],
  [/판막|심내막염|심초음파/i, '승모판,mitral valve,삼첨판,tricuspid valve,대동맥판,aortic valve,폐동맥판,판막협착,판막역류,vegetation,심초음파,echocardiography,Doppler'],
  [/혈압|고혈압|혈관계|죽상동맥|대동맥|말초동맥/i, '혈압,blood pressure,수축기,systole,이완기,diastole,말초저항,동맥,artery,정맥,vein,모세혈관,capillary,죽상경화증,atherosclerosis,내피,endothelium'],
  [/허혈|관상동맥|협심증|심근경색/i, '관상동맥,coronary artery,심근허혈,myocardial ischemia,심근경색,myocardial infarction,협심증,angina,급성관상동맥증후군,ACS,STEMI,NSTEMI,PCI,troponin'],
  [/혈전|색전|응고|출혈/i, '혈전,thrombus,색전,embolus,혈소판,platelet,응고인자,coagulation factor,피브린,fibrin,트롬빈,thrombin,항응고제,anticoagulant,D-dimer'],
  [/연골|뼈|골절|골다공|골조직/i, '연골,cartilage,뼈,bone,연골세포,chondrocyte,골모세포,osteoblast,골세포,osteocyte,파골세포,osteoclast,골기질,osteoid,골막,periosteum,골절,fracture'],
  [/근육|근수축|muscle/i, '골격근,skeletal muscle,근섬유,muscle fiber,근절,sarcomere,액틴,actin,미오신,myosin,트로포닌,troponin,신경근접합부,neuromuscular junction'],
  [/척주|척수|신경|등의|back/i, '척수,spinal cord,척추,vertebra,척수신경,spinal nerve,신경근,nerve root,후근신경절,dorsal root ganglion,피부분절,dermatome,운동신경,감각신경'],
  [/관절|류마티스|인대|힘줄/i, '관절,joint,활막,synovium,관절연골,인대,ligament,힘줄,tendon,류마티스관절염,rheumatoid arthritis,골관절염,osteoarthritis'],
  [/피부/i, '표피,epidermis,진피,dermis,각질세포,keratinocyte,멜라닌세포,melanocyte,피하조직,모낭,hair follicle,피지샘'],
  [/눈|안구|안과|망막|백내장|녹내장|각막|근시|포도막/i, '각막,cornea,수정체,lens,망막,retina,시신경,optic nerve,황반,macula,안압,intraocular pressure,백내장,cataract,녹내장,glaucoma'],
  [/귀|난청|청각|외이|중이|내이/i, '고막,tympanic membrane,중이,middle ear,내이,inner ear,달팽이관,cochlea,청각신경,유모세포,hair cell,감각신경성난청'],
  [/두경부|얼굴|악안면|후두|인두|비부비동|neck|head/i, '두개골,skull,안면신경,facial nerve,삼차신경,trigeminal nerve,인두,pharynx,후두,larynx,경동맥,carotid artery,침샘,salivary gland'],
  [/호흡.*생리|폐기능|가스|산소|환기/i, '폐포,alveolus,환기,ventilation,관류,perfusion,가스교환,gas exchange,확산능,DLCO,일회호흡량,tidal volume,FEV1,FVC,저산소혈증,hypoxemia'],
  [/천식|COPD|폐쇄성|폐기종/i, '천식,asthma,만성폐쇄성폐질환,COPD,폐기종,emphysema,기관지확장제,bronchodilator,흡입스테로이드,FEV1,FVC,기도과민성'],
  [/폐렴|결핵|감염성폐|기관지내시경|객담/i, '폐렴,pneumonia,결핵,tuberculosis,객담,sputum,기관지내시경,bronchoscopy,폐포,alveolus,Mycobacterium tuberculosis,비결핵항산균,NTM'],
  [/간질성폐|폐섬유|호흡곤란|ARDS/i, '호흡곤란,dyspnea,간질성폐질환,interstitial lung disease,폐섬유화,pulmonary fibrosis,급성호흡곤란증후군,ARDS,저산소혈증,hypoxemia'],
  [/흉막|기흉|흉수/i, '흉막,pleura,흉막삼출,pleural effusion,기흉,pneumothorax,흉강천자,thoracentesis,삼출액,exudate,여출액,transudate'],
  [/빈혈|anemia|혈구|hematopoiesis|조혈/i, '조혈,hematopoiesis,적혈구,erythrocyte,백혈구,leukocyte,혈소판,platelet,골수,bone marrow,혈색소,hemoglobin,망상적혈구,reticulocyte,빈혈,anemia'],
  [/백혈병|림프종|myeloid|lymphoid/i, '백혈병,leukemia,림프종,lymphoma,골수,bone marrow,모세포,blast,골수성,myeloid,림프성,lymphoid,면역표현형'],
  [/종양|암|cancer|tumor/i, '종양,tumor,암종,carcinoma,육종,sarcoma,전이,metastasis,생검,biopsy,병기,staging,항암치료,chemotherapy'],
  [/식도|역류|위장|소화성|위염|위암/i, '식도,esophagus,위,stomach,위식도역류,GERD,소화성궤양,peptic ulcer,Helicobacter pylori,내시경,endoscopy,위염,gastritis'],
  [/소장|대장|장내|장질환|크론|흡수|설사/i, '십이지장,duodenum,공장,jejunum,회장,ileum,대장,colon,흡수,absorption,염증성장질환,IBD,크론병,Crohn disease,궤양성대장염,ulcerative colitis'],
  [/간염|간암|간경변|간질환/i, '간세포,hepatocyte,간염,hepatitis,간경변,cirrhosis,문맥고혈압,portal hypertension,HBV,HCV,간세포암,hepatocellular carcinoma,빌리루빈,bilirubin'],
  [/췌장|담낭|담도|담석/i, '췌장,pancreas,췌장염,pancreatitis,담낭,gallbladder,담관,bile duct,담석,gallstone,아밀라아제,amylase,리파아제,lipase'],
  [/upper limb|상지|팔|어깨/i, '상지,upper limb,상완골,humerus,요골,radius,척골,ulna,상완신경총,brachial plexus,정중신경,median nerve,척골신경,ulnar nerve'],
  [/lower limb|하지|다리|무릎/i, '하지,lower limb,대퇴골,femur,경골,tibia,비골,fibula,좌골신경,sciatic nerve,대퇴신경,femoral nerve'],
  [/thorax|흉부|세로칸/i, '흉곽,thorax,종격동,mediastinum,횡격막,diaphragm,늑간신경,intercostal nerve,흉막,pleura'],
  [/abdomen|pelvis|복부|골반/i, '복막,peritoneum,장간막,mesentery,후복막,retroperitoneum,골반,pelvis,복벽,abdominal wall'],
  [/윤리|동의|돌봄|의료분쟁|공정/i, '의료윤리,medical ethics,자율성,autonomy,선행,beneficence,악행금지,nonmaleficence,정의,justice,설명동의,informed consent'],
];
const fallback = { 심순: '심장,heart,순환,circulation,심근,myocardium', 근골: '근골격계,musculoskeletal,뼈,bone,근육,muscle', 호흡: '호흡,respiration,폐,lung,기관지,bronchus', 혈액: '혈액,blood,조혈,hematopoiesis', 소흡: '소화,digestion,흡수,absorption', 해부: '해부학,anatomy', 두경부: '두경부,head and neck', DMH2: '의료윤리,medical ethics', DP2: '진찰,physical examination' };
export function normalizeTerms(value) {
  const unique = new Map();
  for (const raw of (Array.isArray(value) ? value : String(value || '').split(/[,\n]/))) {
    const term = String(raw).normalize('NFC').replace(/[\u0000-\u001f<>|]/g, '').replace(/\s+/g, ' ').trim();
    if (term && term.length <= 100 && !unique.has(term.toLowerCase())) unique.set(term.toLowerCase(), term);
  }
  return [...unique.values()];
}
export function suggestVocabulary(lecture) {
  const title = String(lecture?.title || '').normalize('NFC');
  const matched = topics.filter(([pattern]) => pattern.test(title));
  const titleTerms = title.split(/[+():/]/).map(t => t.trim().replace(/\d+$/, '').trim()).filter(t => t.length > 1 && t.length <= 60 && !/^(과목 OT|실습|강화학습|강화평가|기본평가|발표|Lecture)$/i.test(t));
  const terms = normalizeTerms([...titleTerms, ...matched.flatMap(([, terms]) => terms.split(',')), ...(matched.length ? [] : (fallback[lecture?.course] || '').split(','))]);
  const result = [];
  for (const term of terms) { if (result.length >= 50 || [...result, term].join(', ').length > 1500) break; result.push(term); }
  return result;
}
