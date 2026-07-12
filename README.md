# ml-final-project-walmart-recruiting

## მონაცემების წინასწარი დამუშავება

### NaN მნიშვნელობები

![Missingness chart](assets/missingness.png)

- `MarkDown1`–`MarkDown5` — NaN მნიშვნელობები შეივსება 0-ით, რადგან NaN ნიშნავს რომ მოცემულ კვირაში ფასდაკლება არ ყოფილა.
- `CPI`, `Unemployment` — შეივსება Forward filling / Backward filling მეთოდის გამოყენებით თითოეული მაღაზიისთვის ცალ-ცალკე, რადგან ისინი დროზე დამოკიდებული ცვლადებია.
- `Temperature`, `Fuel_Price` — არ გვაქვს NaN მნიშვნელობები.

## Feature Engineering For All Models

ქვემოთ მოცემული პუნქტები აღწერს როგორ ვამუშავებთ მონაცემებს ყველა მოდელისთვის. თუმცა კონკრეტულ მოდელს შეიძლება კიდევ დამატებით სხვანაირად დასჭირდეს მონაცემთა დამუშავება, ამიტომ ასეთი ტიპის გარდაქმნებს თვითონ ამ მოდელის განხილვის დროს აღვწერთ.

### 1. დროის ცვლადები

ვინაიდან საცალო ვაჭრობა მკვეთრად სეზონურია, ყველა მოდელისთვის სასარგებლოა იმის ცოდნა, კვირა წლის რომელ მონაკვეთში ხვდება. გაყიდვებზე დიდ გავლენას ახდენს დროის პერიოდში არსებული მნიშვნელოვანი მოვლენები (Holidays - ახალი წელი, მადლიერების დღე და ა.შ.)

**დამატებული სვეტები (`add_time_features`):**

იმისათვის რომ თითოეულ მაღაზიაზე და დეპარტამენტზე მოცემული თარიღები მოდელისთვის უფრო აღქმადი იყოს დავამატეთ ცვლადები:
- `Year`, `Month`, `WeekOfYear` - თარიღის ძირითადი დაშლა რიცხვით კომპონენტებად.

საინტერესოა იმ ფაქტის ცოდნა თუ რამდენად ახლოს ვართ რომელიმე მნიშვნელოვან მოვლენასთან. მაგალითად გაყიდვები იზრდება ახალი წლისთვის სამზადისის პერიოდში. ეს პერიოდი გავლენას ახდენს გაყიდვებზე თუმცა მხოლოდ `IsHoliday` ცვლადით ვერ გავითვალისწინებდით მათ. ამის გამო გადავწყვიტეთ დავამატოთ ცვლადები:

- `DaysSinceLastHoliday` - რამდენი დღე გავიდა ბოლო მნიშვნელოვანი მოვლენიდან. 
- `DaysToNextHoliday` - რამდენი დღე დარჩა შემდეგ მნიშვნელოვან მოვლენამდე. 

### 2. მაღაზიის მონაცემები

**დამატებული სვეტები (`add_store_features`):**

- `Type_A`, `Type_B`, `Type_C` - მაღაზიის ტიპის One-hot encoding-ის გამოყენებით, რადგან 3 ტიპი გვაქვს მაღაზიებისთვის და მხოლოდ 3 boolean სვეტის დამატება გვიწევს.
- `Size` - მაღაზიის ფართობი.

### 3. IsHoliday კოდირება

**(`encode_is_holiday`):**

`IsHoliday` სვეტი თავდაპირველად Boolean არის (`True`/`False`). ყველა მოდელისთვის გარდავქმნით მას მთელ რიცხვად (`1`/`0`), რათა მოდელმა შეძლოს მისი გამოყენება.

### 4. features.csv-ის დამერჯვა

**(`merge_features`):**

`features.csv` იმერჯება train და test მონაცემებთან `Store` და `Date` სვეტების მიხედვით, რათა თითოეულ სტრიქონს დაემატოს `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` და `MarkDown1`–`MarkDown5`.


## DLinear მოდელი

DLinear გამოვიყენეთ როგორც time-series forecasting მოდელი, რომელიც კარგად ერგება Walmart-ის ტიპის weekly sales forecasting ამოცანას. მონაცემები გადავიყვანეთ `NeuralForecast`-ის long format-ში, რადგან DLinear თითოეულ Store-Dept სერიას ცალკე time series-ად ხედავს:

- `unique_id` - ერთი time-series თითოეული `Store` + `Dept` წყვილისთვის.
- `ds` - კვირის თარიღი.
- `y` - სამიზნე ცვლადი, ანუ `Weekly_Sales`.

DLinear-ის მთავარი იდეაა, რომ time-series იყოფა ორ ნაწილად: trend კომპონენტად და seasonal/remainder კომპონენტად. შემდეგ ორივე კომპონენტზე გამოიყენება მარტივი linear projection, რომელიც ბოლო ისტორიული ფანჯრიდან პროგნოზირებს მომავალ კვირებს. ეს არქიტექტურა ბევრად უფრო მარტივია, ვიდრე დიდი recurrent ან attention-based მოდელები, მაგრამ ძლიერი baseline არის ისეთი მონაცემებისთვის, სადაც ისტორიული გაყიდვების pattern-ები, სეზონურობა და store-department-ის სპეციფიკური ჩვევები ძალიან მნიშვნელოვანია.

### რატომ ვიყენებთ მხოლოდ ისტორიულ ინფორმაციას

DLinear-ის ამ ექსპერიმენტში მოდელს ვაწვდით მხოლოდ ისტორიულ `Weekly_Sales` მნიშვნელობებს. ანუ, თითოეული `Store-Dept` სერიისთვის მოდელი იღებს წარსული 52 კვირის გაყიდვებს და ამ history-ზე დაყრდნობით პროგნოზირებს შემდეგ 39 კვირას.

DLinear-ის არქიტექტურა ძალიან მარტივია: ის input window-ს შლის trend და seasonal ნაწილებად და შემდეგ linear layer-ების საშუალებით ასახავს ამას მომავალ horizon-ზე. ამ მოდელში არ გვაქვს ცალკე decoder ან attention მექანიზმი, სადაც მომავალ კვირებზე ცნობილი ცვლადები ცალკე შევიდოდა. ამიტომ DLinear-ისთვის ყველაზე ბუნებრივი setup არის history-based forecasting: მოდელი სწავლობს წარსული გაყიდვების pattern-ებს და მომავალის პრედიქციას ახდენს.

ეს მიდგომა სწორია forecast ამოცანისთვის, რადგან validation და test პროგნოზის დროს მოდელმა უნდა გამოიყენოს მხოლოდ ის ინფორმაცია, რომელიც პროგნოზის მომენტში რეალურად ხელმისაწვდომია. 

ამიტომ DLinear-ის საბოლოო ვერსია არის `target_history_only`: input-ში გვაქვს მხოლოდ `unique_id`, `ds` და `y`. ეს არ ნიშნავს, რომ სხვა features უსარგებლოა, უბრალოდ ამ მოდელის მიზანია გვაჩვენოს, რამდენად ძლიერი პროგნოზი შეიძლება მივიღოთ მხოლოდ historical sales pattern-ებიდან.

### Train/Validation setup

DLinear შევაფასეთ time-based validation-ით. მოდელი ვავარჯიშეთ ძველ კვირების ინფორმაციაზე და შემდეგ, ახალ კვირებზე შევამოწმებთ. ასეთი split აუცილებელია forecasting ამოცანაში, რადგან რეალურ ცხოვრებაშიც წარსულით ვცდილობთ მომავლის პროგნოზირებას.

validation setup:

- Train პერიოდი: `2010-02-05`-დან `2012-01-27`-მდე
- Validation პერიოდი: `2012-02-03`-დან `2012-10-26`-მდე
- Input window: `52` კვირა, ანუ მოდელი ყოველი პროგნოზისთვის უყურებს ბოლო ერთ წელს
- Forecast horizon: `39` კვირა, რაც ემთხვევა test set-ის კვირების რაოდენობას
- Frequency: weekly Friday (`W-FRI`)


მონაცემებში ყველა `Store-Dept` სერიას ერთნაირი რაოდენობის კვირები არ ჰქონდა. DLinear-ის cross-validation რომ სტაბილურად გაშვებულიყო, train/evaluation ნაწილში გამოვიყენეთ სრული ისტორიის მქონე სერიები:

- სულ Store-Dept time series: `3331`
- სრული ისტორიის მქონე რიგები: `2660`
- მოკლე ან არათანაბარი სერიები, რომლებიც DLinear train/evaluation-იდან ამოვიღეთ: `671`

საბოლოო prediction pipeline-ში მოკლე სერიებისთვის fallback ლოგიკაც დავამატეთ. თუ DLinear კონკრეტულ `Store-Dept` წყვილზე პროგნოზს ვერ აბრუნებს, ვიყენებთ ამ სერიის ბოლო ცნობილ `Weekly_Sales` მნიშვნელობას. თუ არც ეს არსებობს, ვიყენებთ გლობალურ median fallback-ს (`7,612.03`).

### Hyperparameter search

გავუშვით DLinear-ის რამდენიმე configuration და ისინი დავყავით underfit/balanced/overfit.

ასეთი დაყოფა დაგვეხმარა გვენახა, როგორ რეაგირებს DLinear სხვადასხვა სირთულის setup-ზე. Underfit configuration-ები გვაჩვენებს შემთხვევებს, სადაც მოდელი ზედმეტად მარტივია და historical pattern-ებს საკმარისად ვერ სწავლობს. Overfit configuration-ები პირიქით, გვაჩვენებს შემთხვევებს, სადაც მოდელი train მონაცემს ზედმეტად ერგება და validation-ზე უარესად მუშაობს. Balanced configuration-ების მიზანი იყო ამ ორ უკიდურესობას შორის უკეთესი trade-off-ის პოვნა, სადაც validation WMAE ყველაზე დაბალია.

ამ შედარებამ გვაჩვენა, რომ DLinear-ისთვის საუკეთესო შედეგი არ მოდის უბრალოდ უფრო დიდი ან უფრო ხანგრძლივად ნავარჯიშები მოდელიდან. უკეთესი შედეგი მივიღეთ მაშინ, როცა input window, moving average window და training steps ერთმანეთთან დაბალანსებული იყო.

ეს ბალანსი შემთხვევით არ აგვირჩევია. საუკეთესო configuration იყო ის, სადაც მოდელს საკმარისი ისტორია ჰქონდა სასარგებლო pattern-ების დასასწავლად, მაგრამ არც ისე დიდი complexity ან training steps, რომ train set-ზე ზედმეტად მორგებულიყო.

ძირითადი tuning parameters იყო:

- `input_size`
- `moving_avg_window`
- `max_steps`
- `learning_rate`
- `batch_size`

საუკეთესო run იყო `balanced_2`:

| პარამეტრი | მნიშვნელობა |
|---|---:|
| `input_size` | `52` |
| `moving_avg_window` | `13` |
| `max_steps` | `500` |
| `learning_rate` | `0.001` |
| `batch_size` | `128` |
| Validation WMAE | `2,555.44` |

WMAE გამოვიყენეთ როგორც მთავარი metric, რადგან Walmart-ის competition-ის შეფასებაშიც holiday weeks უფრო მაღალი წონით ფასდება. ეს მნიშვნელოვანია, რადგან holiday periods გაყიდვებზე ძლიერ გავლენას ახდენს და ასეთ კვირებში მოდელის შეცდომა უფრო დიდ გავლენას ახდენს საბოლოო შეფასებაზე.

### DLinear plots

ქვემოთ მოცემული plot აჩვენებს DLinear runs-ის შედარებას validation WMAE-ის მიხედვით. მთავარი მიზანი იყო გვეპოვა ის hyperparameter configuration, რომელსაც held-out validation პერიოდზე ყველაზე დაბალი შეცდომა ჰქონდა. საუკეთესო შედეგი მიიღო `balanced_2` configuration-მა.

<img src="notebooks/Deep%20Learning/Plots/dlinear_wmae_comparison.png" alt="DLinear WMAE comparison" width="600">

შემდეგი plot აჩვენებს იმ Store-Dept წყვილებს, სადაც validation error ყველაზე მაღალი იყო. ყველაზე რთული სერიები აღმოჩნდა, მაგალითად, `(10, 72)`, `(14, 92)`, `(20, 72)`, `(35, 72)` და `(18, 92)`. ასეთი error analysis მნიშვნელოვანია, რადგან overall WMAE კარგ სურათს გვაძლევს, მაგრამ კონკრეტული პრობლემური departments აჩვენებს სად შეიძლება დაგვჭირდეს დამატებითი feature engineering ან სხვა მოდელის გამოყენება.

<img src="notebooks/Deep%20Learning/Plots/dlinear_worst_store_dept.png" alt="DLinear worst Store-Dept validation errors" width="600">

Holiday vs non-holiday error-იც რომ შევადაროთ:

- Non-holiday MAE: `2,565.99`
- Holiday MAE: `2,516.43`

ეს ნიშნავს, რომ DLinear-ს holiday კვირებზე არ ჰქონია მკვეთრად უარესი performance. ასეთი შედეგი ლოგიკურია, რადგან historical `Weekly_Sales` უკვე შეიცავს recurring holiday spikes-ს და DLinear-ს შეუძლია ამ pattern-ის ნაწილის დაჭერა მხოლოდ target history-დანაც.

### დასკვნა

DLinear ამ პროექტში გამოვიყენეთ როგორც history-based forecasting baseline. მისი მიზანი იყო გვენახა, რამდენად კარგად შეგვიძლია Walmart-ის weekly sales-ის პროგნოზირება მხოლოდ წარსული გაყიდვების დინამიკით, დამატებითი exogenous features-ის გარეშე.

საუკეთესო DLinear configuration-მა validation-ზე მიიღო `2,555.44` WMAE. ეს შედეგი აჩვენებს, რომ historical `Weekly_Sales` უკვე შეიცავს ბევრ მნიშვნელოვან სიგნალს: სეზონურობას, holiday uplift-ს, department-specific behavior-ს და store-level demand pattern-ებს. ანუ, მიუხედავად იმისა, რომ მოდელი არ იყენებს `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` ან `MarkDown` features-ს, მხოლოდ გაყიდვების history-დან მაინც შეუძლია ძლიერი პროგნოზის გაკეთება.

DLinear-ის მთავარი უპირატესობა მისი სიმარტივეა. მოდელი სწრაფად train-დება, მარტივად კონტროლდება და კარგი benchmark-ია უფრო რთულ მოდელებთან შედარებისთვის. თუ უფრო კომპლექსური მოდელი DLinear-ზე უკეთეს შედეგს ვერ აჩვენებს, მაშინ დამატებითი სირთულე შეიძლება არც გვიღირდეს.

## N-BEATS მოდელი

N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) გამოვიყენეთ როგორც უფრო ძლიერი deep learning ალტერნატივა DLinear-ის შემდეგ. DLinear-ის მსგავსად, მონაცემები გადავიყვანეთ `NeuralForecast`-ის long format-ში, სადაც თითოეული `Store-Dept` წყვილი ცალ-ცალკე time series-ად განიხილება:

- `unique_id` — ერთი time-series თითოეული `Store` + `Dept` წყვილისთვის.
- `ds` — კვირის თარიღი.
- `y` — სამიზნე ცვლადი, ანუ `Weekly_Sales`.

N-BEATS-ის მთავარი იდეაა, რომ ქსელი დაყოფილია სტეკებად — **trend** სტეკი და **seasonality** სტეკი. თითოეული სტეკი შედგება რამდენიმე ბლოკისგან, სადაც ყოველი ბლოკი წარმოქმნის ორ გამოსავალს: **backcast** (input ფანჯრის რეკონსტრუქცია) და **forecast** (მომავლის პროგნოზი). Backcast-ი იმავე სტეკის შემდეგ ბლოკს გამოაკლდება, რათა ბლოკებმა ნარჩენები ისწავლონ. Trend სტეკი პოლინომიურ ფუნქციებს იყენებს გრძელვადიანი ტენდენციების დასაჭერად, ხოლო Seasonality სტეკი — Fourier ფუნქციებს განმეორებადი კანონზომიერებებისთვის. ყველა სტეკის forecast-ები ჯამდება საბოლოო პროგნოზში.

ეს სტრუქტურა N-BEATS-ს DLinear-თან შედარებით უფრო ინტერპრეტირებადს ხდის: trend სტეკიდან ვხედავთ გრძელვადიან ზრდა/კლების ტენდენციას, ხოლო seasonality სტეკიდან — კვირობრივ და წლიურ სეზონურ pattern-ებს.

### რატომ ვიყენებთ მხოლოდ ისტორიულ ინფორმაციას

N-BEATS-ის ამ ექსპერიმენტში მოდელს ვაწვდით მხოლოდ ისტორიულ `Weekly_Sales` მნიშვნელობებს. თითოეული `Store-Dept` სერიისთვის მოდელი იღებს წარსული 52 კვირის გაყიდვებს და ამ history-ზე დაყრდნობით პროგნოზირებს შემდეგ 39 კვირას.

ეს მიდგომა სწორია, რადგან historical `Weekly_Sales` უკვე შეიცავს recurring holiday spikes-ს, სეზონურ კანონზომიერებებს და store-department-ის სპეციფიკურ ქცევებს. მიზანია გვენახა, რამდენად შეუძლია N-BEATS-ს ამ ყველაფრის ავტომატურად გამოყოფა მხოლოდ target history-დან.

### Train/Validation setup

N-BEATS შევაფასეთ DLinear-ის იდენტური time-based validation სქემით, რათა შედეგები პირდაპირ შედარებადი ყოფილიყო.

validation setup:

- Train პერიოდი: `2010-02-05`-დან `2012-01-27`-მდე
- Validation პერიოდი: `2012-02-03`-დან `2012-10-26`-მდე
- Input window: `52` კვირა, ანუ მოდელი ყოველი პროგნოზისთვის უყურებს ბოლო ერთ წელს
- Forecast horizon: `39` კვირა, რაც ემთხვევა test set-ის კვირების რაოდენობას
- Frequency: weekly Friday (`W-FRI`)

DLinear-ის მსგავსად, გამოვიყენეთ სრული ისტორიის მქონე სერიები:

- სულ Store-Dept time series: `3331`
- სრული ისტორიის მქონე რიგები: `2660`
- მოკლე ან არათანაბარი სერიები, რომლებიც N-BEATS train/evaluation-იდან ამოვიღეთ: `671`

საბოლოო prediction pipeline-ში მოკლე სერიებისთვის fallback ლოგიკა დავამატეთ. თუ N-BEATS კონკრეტულ `Store-Dept` წყვილზე პროგნოზს ვერ აბრუნებს, ვიყენებთ ამ სერიის ბოლო ცნობილ `Weekly_Sales` მნიშვნელობას. თუ არც ეს არსებობს, ვიყენებთ გლობალურ median fallback-ს (`7,612.03`).

Baseline N-BEATS run-მა (30 configs-მდე sweep-ის გარეშე) მიაღწია:

- Train WMAE: `10,280.63`
- Validation WMAE: `1,922.20`

### Hyperparameter search

გავუშვით N-BEATS-ის 30 configuration და ისინი დავყავით underfit / balanced / overfit კატეგორიებად.

ეს დაყოფა გვეხმარება გვენახა, როგორ რეაგირებს N-BEATS სხვადასხვა სირთულის setup-ზე. Underfit configuration-ები გვაჩვენებს შემთხვევებს, სადაც სტეკები ძალიან პატარაა და სერიების complexity-ს ვერ ასახავს. Overfit configuration-ები — სადაც ბლოკები ძალიან ღრმაა ან MLP ძალიან ფართო, train set-ზე ზედმეტად ხდება მორგება. Balanced configuration-ების მიზანი იყო ამ ორ უკიდურესობას შორის საუკეთესო trade-off-ის პოვნა.

N-BEATS-ისთვის საკვანძო Fourier-based seasonality სტეკის `n_harmonics` და polynomial trend სტეკის `n_polynomials` შერჩევა განსაკუთრებით მნიშვნელოვანია. ზედმეტად დიდი `n_harmonics` ან `n_polynomials` ამატებს flexibility-ს, მაგრამ ზრდის overfit-ის რისკსაც.

ძირითადი tuning parameters იყო:

- `input_size`
- `stack_types`
- `n_blocks`
- `mlp_units`
- `n_harmonics`
- `n_polynomials`
- `max_steps`
- `learning_rate`
- `batch_size`

საუკეთესო run იყო `balanced_9`:

| პარამეტრი | მნიშვნელობა |
|---|---:|
| `input_size` | `52` |
| `stack_types` | `['trend', 'seasonality']` |
| `n_blocks` | `[3, 3]` |
| `mlp_units` | `[[512, 512], [512, 512]]` |
| `n_harmonics` | `4` |
| `n_polynomials` | `3` |
| `max_steps` | `800` |
| `learning_rate` | `0.0005` |
| `batch_size` | `128` |
| Validation WMAE | `1,858.23` |

WMAE გამოვიყენეთ როგორც მთავარი metric, რადგან Walmart-ის competition-ის შეფასებაშიც holiday weeks უფრო მაღალი წონით ფასდება. ეს მნიშვნელოვანია, რადგან holiday periods გაყიდვებზე ძლიერ გავლენას ახდენს და ასეთ კვირებში მოდელის შეცდომა უფრო დიდ გავლენას ახდენს საბოლოო შეფასებაზე.

### N-BEATS plots

ქვემოთ მოცემული plot აჩვენებს N-BEATS runs-ის შედარებას validation WMAE-ის მიხედვით. მთავარი მიზანი იყო გვეპოვა ის hyperparameter configuration, რომელსაც held-out validation პერიოდზე ყველაზე დაბალი შეცდომა ჰქონდა.

<img src="notebooks/Deep%20Learning/Plots/nbeats_wmae_comparison.png" alt="N-BEATS WMAE comparison" width="600">

### დასკვნა

N-BEATS ამ პროექტში გამოვიყენეთ როგორც interpretable deep learning მოდელი, რომელიც ავტომატურად ყოფს time series-ს trend და seasonality კომპონენტებად. მისი მიზანი იყო გვენახა, შეუძლია თუ არა სტრუქტურულ decomposition-ზე დამყარებულ neural ქსელს DLinear-ზე მეტი სიგნალის ამოღება ისტორიული გაყიდვების history-დან.

საუკეთესო N-BEATS configuration-მა (`balanced_9`) validation-ზე მიიღო `1,858.23` WMAE, რაც DLinear-ის საუკეთეს შედეგს (`2,555.44`) საგრძნობლად აჯობა. ეს სხვაობა გვიჩვენებს, რომ trend/seasonality decomposition-ის სტრუქტურა და ბლოკ-სტეკური architecture — სადაც ყოველი ბლოკი ნარჩენებს ამუშავებს — Walmart-ის სეზონური weekly sales pattern-ებისთვის ბევრად უფრო შესაფერისია, ვიდრე მარტივი linear projection.

N-BEATS-ის მთავარი უპირატესობა მისი interpretability-ია: trend სტეკი გვაჩვენებს გრძელვადიანი ზრდა/კლების ტენდენციას, ხოლო seasonality სტეკი — განმეორებად კვირობრივ და წლიურ pattern-ებს. ეს insight-ი პრაქტიკული მნიშვნელობა აქვს — შეგვიძლია გვესმოდეს, სად ჭირდება მოდელს გაუმჯობესება და რა ტიპის pattern-ებს ვერ ჭერს.

## Temporal Fusion Transformer (TFT) მოდელი

TFT გამოვიყენეთ როგორც attention-based deep learning მოდელი, რომელიც DLinear-ისა და N-BEATS-ისგან განსხვავებით გათვლილია exogenous features-ის გათვალისწინებაზე. მონაცემები გადავიყვანეთ `NeuralForecast`-ის long format-ში, თითოეული `Store-Dept` წყვილი ცალ-ცალკე time series-ად:

- `unique_id` — ერთი time-series თითოეული `Store` + `Dept` წყვილისთვის.
- `ds` — კვირის თარიღი.
- `y` — სამიზნე ცვლადი, ანუ `Weekly_Sales`.

TFT-ის მთავარი იდეაა გააერთიანოს სამი ტიპის ინფორმაცია: **static covariates** (მაღაზიის ტიპი, ზომა, Store/Dept ID), **known future covariates** (holidays, time features, macro indicators), და **past observed target** (ისტორიული გაყიდვები). Variable Selection Network ირჩევს რომელი feature-ია ყველაზე სასარგებლო თითოეული სერიისთვის, Gated Residual Networks კი ფილტრავს ნაკლებად სასარგებლო სიგნალებს. Temporal Self-Attention mechanism-ი სერიაში შორეულ კვირებს შორის კავშირებს ახდენს, რაც საშუალებას აძლევს მოდელს გამოიყენოს, მაგალითად, გასული წლის holiday spike-ები ამ წლის პროგნოზისთვის.

### რატომ ვიყენებთ exogenous features-ს

TFT-ის ამ ექსპერიმენტში პირველად გამოვიყენეთ სრული feature set — ისტორიული გაყიდვებისა და exogenous ცვლადების კომბინაცია. TFT-ის არქიტექტურა სპეციალურად შექმნილია კოვარიანტების ორ კატეგორიასთან სამუშაოდ:

**Future exogenous** (`15` სვეტი) — ცვლადები, რომლებიც მომავალ კვირებზეც ცნობილია პროგნოზის დროს:
`IsHoliday`, `Temperature`, `Fuel_Price`, `MarkDown1`–`MarkDown5`, `CPI`, `Unemployment`, `Year`, `Month`, `WeekOfYear`, `DaysSinceLastHoliday`, `DaysToNextHoliday`

**Static exogenous** (`6` სვეტი) — ცვლადები, რომლებიც დროში არ იცვლება:
`Store`, `Dept`, `Size`, `Type_A`, `Type_B`, `Type_C`

ეს feature set განსხვავდება DLinear-ისა და N-BEATS-ის `target_history_only` მიდგომისგან. TFT-ის მიზანი იყო გვენახა, შეუძლია თუ არა მოდელს exogenous სიგნალებით (holiday dates, markdowns, macro) უფრო მეტი სიზუსტის მიღება.

### Train/Validation setup

TFT შევაფასეთ DLinear-ისა და N-BEATS-ის იდენტური time-based validation სქემით, რათა შედეგები პირდაპირ შედარებადი ყოფილიყო.

validation setup:

- Train პერიოდი: `2010-02-05`-დან `2012-01-27`-მდე
- Validation პერიოდი: `2012-02-03`-დან `2012-10-26`-მდე
- Input window: `52` კვირა, ანუ მოდელი ყოველი პროგნოზისთვის უყურებს ბოლო ერთ წელს
- Forecast horizon: `26` კვირა
- Frequency: weekly Friday (`W-FRI`)

სხვა მოდელების მსგავსად, გამოვიყენეთ სრული ისტორიის მქონე სერიები:

- სულ Store-Dept time series: `3331`
- სრული ისტორიის მქონე რიგები: `2660`
- მოკლე ან არათანაბარი სერიები, რომლებიც TFT train/evaluation-იდან ამოვიღეთ: `671`

საბოლოო prediction pipeline-ში მოკლე სერიებისთვის fallback ლოგიკა დავამატეთ. თუ TFT კონკრეტულ `Store-Dept` წყვილზე პროგნოზს ვერ აბრუნებს, ვიყენებთ ამ სერიის ბოლო ცნობილ `Weekly_Sales` მნიშვნელობას. თუ არც ეს არსებობს, ვიყენებთ გლობალურ median fallback-ს (`7,612.03`).

### Hyperparameter search

გავუშვით TFT-ის 16 configuration (30-დან — overfit configs გამოტოვდა compute constraints-ის გამო) და ისინი დავყავით underfit / balanced კატეგორიებად.

Underfit configuration-ები გვაჩვენებს შემთხვევებს, სადაც hidden size ძალიან პატარაა ან training steps ძალიან ცოტაა და attention mechanism-ს საკმარისი სიღრმე არ აქვს სასარგებლო კავშირების სასწავლად. Balanced configuration-ების მიზანი იყო hidden size, n_head, dropout და training steps-ის ოპტიმალური კომბინაციის პოვნა.

TFT-ისთვის განსაკუთრებით მნიშვნელოვანია `hidden_size` (embedding სივრცის სიგანე) და `n_head` (attention head-ების რაოდენობა) — ეს ორი პარამეტრი განსაზღვრავს, რამდენ parallel pattern-ს სწავლობს მოდელი ერთდროულად.

ძირითადი tuning parameters იყო:

- `input_size`
- `hidden_size`
- `n_head`
- `dropout`
- `max_steps`
- `learning_rate`
- `batch_size`

საუკეთესო run იყო `balanced_1`:

| პარამეტრი | მნიშვნელობა |
|---|---:|
| `input_size` | `52` |
| `hidden_size` | `32` |
| `n_head` | `2` |
| `dropout` | `0.10` |
| `max_steps` | `300` |
| `learning_rate` | `0.001` |
| `batch_size` | `128` |
| Validation WMAE | `2,216.19` |

WMAE გამოვიყენეთ როგორც მთავარი metric, რადგან Walmart-ის competition-ის შეფასებაშიც holiday weeks უფრო მაღალი წონით ფასდება.

### TFT plots

ქვემოთ მოცემული plot აჩვენებს TFT runs-ის შედარებას validation WMAE-ის მიხედვით. მთავარი მიზანი იყო გვეპოვა ის hyperparameter configuration, რომელსაც held-out validation პერიოდზე ყველაზე დაბალი შეცდომა ჰქონდა. საუკეთესო შედეგი მიიღო `balanced_1` configuration-მა.

<img src="notebooks/Deep%20Learning/Plots/tft_wmae_comparison.png" alt="TFT WMAE comparison" width="600">

შემდეგი plot აჩვენებს იმ Store-Dept წყვილებს, სადაც validation error ყველაზე მაღალი იყო. ყველაზე რთული სერიები აღმოჩნდა `(14, 92)`, `(10, 72)`, `(14, 95)`, `(28, 92)` და `(14, 72)`.

<img src="notebooks/Deep%20Learning/Plots/tft_worst_store_dept.png" alt="TFT worst Store-Dept validation errors" width="600">

Holiday vs non-holiday error-იც რომ შევადაროთ:

- Non-holiday MAE: `2,079.36`
- Holiday MAE: `2,722.47`

ეს ნიშნავს, რომ TFT-ს holiday კვირებზე საგრძნობლად უფრო მაღალი შეცდომა ჰქონდა. ეს მოსალოდნელია, რადგან holiday periods-ში გაყიდვები კომპლექსური spike-ებს ქმნის, რომლებიც ყოველ წელს სხვადასხვა ინტენსივობისაა. მიუხედავად იმისა, რომ TFT-ს `IsHoliday` და `DaysToNextHoliday` features-ები ჰქონდა, ეს spike-ების ზუსტი სიდიდე history-ზე დაყრდნობით ძნელი სასწავლია.

### დასკვნა

TFT ამ პროექტში გამოვიყენეთ როგორც პირველი მოდელი, რომელიც სრულ exogenous feature set-ს იყენებს — future covariates-სა და static covariates-ს. მისი მიზანი იყო გვენახა, შეუძლია თუ არა attention mechanism-სა და variable selection-ზე დამყარებულ მოდელს target history-ს მიღმა სიგნალებით შედეგის გაუმჯობესება.

საუკეთესო TFT configuration-მა (`balanced_1`) validation-ზე მიიღო `2,216.19` WMAE. ეს შედეგი DLinear-ს (`2,555.44`) აჯობა, მაგრამ N-BEATS-ს (`1,858.23`) ჩამოუვარდა, მიუხედავად იმისა, რომ TFT-ს გაცილებით მეტი ინფორმაცია ჰქონდა (15 future + 6 static features). ეს გვიჩვენებს, რომ ამ მონაცემებში სეზონური history-ს პირდაპირი decomposition უფრო ძლიერი სიგნალია, ვიდრე exogenous features-ის attention-based კომბინაცია.

TFT-ის მთავარი უპირატესობა მისი flexibility-ია: Variable Selection Network-ი ავტომატურად ათეულობით feature-დან ირჩევს ყველაზე სასარგებლოს. ეს განსაკუთრებით ღირებულია Walmart-ის მონაცემებში, სადაც `MarkDown` features-ი მხოლოდ გარკვეულ Store-Dept წყვილებზე მოქმედებს.

## LightGBM მოდელი

LightGBM გამოვიყენეთ როგორც gradient boosting მოდელი, რომელიც ყველა წინა მოდელისგან პრინციპულად განსხვავდება: ის არ არის per-series მოდელი. ARIMA, DLinear, N-BEATS და TFT თითოეული Store-Dept სერიაზე ცალ-ცალკე მუშაობდა, ხოლო LightGBM ერთი გლობალური tabular მოდელია, რომელიც მთელ train set-ს ერთდროულად ხედავს. `Store` და `Dept` მოდელს categorical feature-ებად გადაეცემა, ამიტომ მოდელს შეუძლია cross-series pattern-ების სწავლა — მაგ., რომ Dept 72 ყოველ წელს holiday spike-ს ქმნის ყველა მაღაზიაში.

### feature set — `full_features_plus_safe_lag`

LightGBM-ი ამ პროექტში პირველი მოდელია, რომელიც ერთდროულად იყენებს:

**Exogenous features:**
`Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `MarkDown1`–`MarkDown5`, `Type_A/B/C`, `Size`, `IsHoliday`, `DaysSinceLastHoliday`, `DaysToNextHoliday`, `Year`, `Month`, `WeekOfYear`

**Lag features (leakage-safe):**
- `lag_52` — გასული წლის იმავე კვირის გაყიდვები. 52-კვირიანი lag ყოველთვის უსაფრთხოა, რადგან forecast horizon მაქსიმუმ 39 კვირაა.
- `roll_mean_26_lag52` — 26-კვირიანი rolling mean, lag-52 წერტილიდან დათვლილი.
- `roll_std_26_lag52` — 26-კვირიანი rolling std, lag-52 წერტილიდან.

**Store/Dept identifiers:** `Store`, `Dept`

Rolling stats lag-52 წერტილიდან ითვლება (არა მიმდინარე თარიღიდან), რათა test-time-ზე lag-ები ყოველთვის ხელმისაწვდომი იყოს.

### Train/Validation setup

LightGBM შევაფასეთ სხვა მოდელების იდენტური time-based validation სქემით.

validation setup:

- Train პერიოდი: `2010-02-05`-დან `2012-01-27`-მდე
- Validation პერიოდი: `2012-02-03`-დან `2012-10-26`-მდე
- Forecast horizon: `26` კვირა
- Frequency: weekly Friday (`W-FRI`)

LightGBM გლობალური მოდელია, ამიტომ სერიების გაფილტვრა არ გვჭირდება — მოდელი ყველა `3331` Store-Dept წყვილს ერთ train set-ში ხედავს.

Baseline LightGBM run-მა მიაღწია:

- Train WMAE: `2,326.38`
- Validation WMAE: `1,864.71`

### Hyperparameter search

გავუშვით LightGBM-ის hyperparameter sweep, თუმცა runtime constraints-ის გამო sweep-ი შეწყდა პირველი კონფიგურაციის შემდეგ. ამიტომ baseline configuration გამოვიყენეთ საბოლოო მოდელად.

ძირითადი tuning parameters იყო:

- `num_leaves`
- `max_depth`
- `n_estimators`
- `learning_rate`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `reg_alpha`, `reg_lambda`

Baseline კონფიგურაცია:

| პარამეტრი | მნიშვნელობა |
|---|---:|
| `num_leaves` | `31` |
| `max_depth` | `-1` (unlimited) |
| `n_estimators` | `300` |
| `learning_rate` | `0.05` |
| `min_child_samples` | `20` |
| `subsample` | `0.8` |
| `colsample_bytree` | `0.8` |
| Validation WMAE | `1,864.71` |

### LightGBM plots

ქვემოთ მოცემული plot აჩვენებს LightGBM runs-ის შედარებას validation WMAE-ის მიხედვით.

<img src="notebooks/Plots/lightgbm_wmae_comparison.png" alt="LightGBM WMAE comparison" width="600">

Feature importance plot გვიჩვენებს, რომელ feature-ებს LightGBM ყველაზე მეტ გამოყენებას უკეთებდა პროგნოზის დასამზადებლად.

<img src="notebooks/Plots/lightgbm_feature_importance.png" alt="LightGBM feature importance" width="600">

შემდეგი plot აჩვენებს იმ Store-Dept წყვილებს, სადაც validation error ყველაზე მაღალი იყო.

<img src="notebooks/Plots/lightgbm_worst_store_dept.png" alt="LightGBM worst Store-Dept validation errors" width="600">

### დასკვნა

LightGBM ამ პროექტში გამოვიყენეთ როგორც გლობალური tabular მოდელი — ერთი მოდელი მთელი მონაცემების მართვისთვის. მისი მიზანი იყო გვენახა, შეიძლება თუ არა cross-series pattern-ებისა და exogenous features-ის კომბინაციით per-series მოდელებთან კონკურენცია.

Baseline LightGBM-მა validation-ზე მიიღო `1,864.71` WMAE, რაც TFT-ს (`2,216.19`) და DLinear-ს (`2,555.44`) საგრძნობლად აჯობა და N-BEATS-ის (`1,858.23`) თითქმის გაუტოლდა — მხოლოდ `6.48`-ით ჩამოუვარდა. ეს შედეგი მნიშვნელოვანია, რადგან LightGBM ბევრად სწრაფად train-დება ვიდრე ნებისმიერი neural network და hyperparameter-ების sweep-ის გარეშეც კი ძლიერ შედეგს იძლევა.

LightGBM-ის მთავარი უპირატესობა სიჩქარე და სტაბილურობაა: სრული train set-ი წუთებში დამუშავდება, feature importance გამჭვირვალეა და მოდელი lag features-ს გლობალურ cross-series pattern-ებთან ერთად ეფექტურად ითვისებს.

## XGBoost მოდელი

XGBoost გამოვიყენეთ როგორც მეორე gradient boosting მოდელი LightGBM-ის გვერდით. LightGBM-ის მსგავსად, XGBoost ერთი გლობალური tabular მოდელია, რომელიც მთელ train set-ს ერთდროულად ამუშავებს — `Store` და `Dept` numeric feature-ებად გადაეცემა მოდელს. ორივე მოდელის შედეგები პირდაპირ შედარებადია, რადგან იდენტური feature set და validation split გამოვიყენეთ.

### feature set — `full_exogenous`

XGBoost-ი იყენებს ყველა ხელმისაწვდომ feature-ს, მათ შორის სპეციალურ lag features-ს, რომლებიც XGBoost-ისთვის ცალკე გამოვთვალეთ:

**Lag features (leakage-safe):**
- `lag_26` — 26 კვირის წინანდელი გაყიდვები (მინიმალური უსაფრთხო lag 26-კვირიანი horizon-ისთვის)
- `lag_52` — გასული წლის იმავე კვირის გაყიდვები

**Rolling features** (lag-26 წერტილიდან დათვლილი, leakage-safe):
- `rolling_mean_4/13/26` — trailing rolling mean
- `rolling_std_4/13/26` — trailing rolling std

**Exogenous features:**
`IsHoliday`, `DaysSinceLastHoliday`, `DaysToNextHoliday`, `Fuel_Price`, `Temperature`, `CPI`, `Unemployment`, `MarkDown1`–`MarkDown5`

**Store features:** `Type_A`, `Type_B`, `Type_C`, `Size`, `Store`, `Dept`

**Calendar features:** `WeekOfYear`, `Month`, `Year`

Holiday კვირები train-ის დროს `sample_weight=5`-ით ფასდება, რათა WMAE metric-თან შესაბამისი bias შეიქმნას.

### Train/Validation setup

XGBoost შევაფასეთ სხვა მოდელების იდენტური time-based validation სქემით.

validation setup:

- Train პერიოდი: `2010-02-05`-დან `2012-01-27`-მდე
- Validation პერიოდი: `2012-02-03`-დან `2012-10-26`-მდე
- Forecast horizon: `26` კვირა
- Frequency: weekly Friday (`W-FRI`)

XGBoost გლობალური მოდელია, ამიტომ სერიების გაფილტვრა არ გვჭირდება. თუმცა lag features-ის გამოთვლის შემდეგ პირველი 52 სტრიქონი თითოეული სერიისთვის იშლება (NaN lag-ების გამო), ამიტომ matrix build-ის დროს ნაწილი მონაცემებისა გამოიყენება.

Baseline XGBoost run-მა მიაღწია:

- Train WMAE: `1,587.36`
- Validation WMAE: `1,800.42`
- Gap: `13.4%` (good)

### Hyperparameter search

გავუშვით XGBoost-ის 30 configuration და ისინი დავყავით underfit / balanced / overfit კატეგორიებად.

XGBoost-ისთვის `max_depth` და `n_estimators` ყველაზე გავლენიანი პარამეტრებია — ხის სიღრმე განსაზღვრავს feature interaction-ების სირთულეს, ხოლო estimator-ების რაოდენობა ბoosting round-ების სიჭარბეს. `learning_rate`-ის შემცირება ჩვეულებრივ მეტ `n_estimators`-ს მოითხოვს.

ძირითადი tuning parameters იყო:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`, `reg_alpha`

საუკეთესო run იყო `overfit_5`:

| პარამეტრი | მნიშვნელობა |
|---|---:|
| `n_estimators` | `2000` |
| `max_depth` | `8` |
| `learning_rate` | `0.005` |
| `subsample` | `0.9` |
| `colsample_bytree` | `0.9` |
| `min_child_weight` | `1` |
| `reg_lambda` | `0.1` |
| `reg_alpha` | `0.0` |
| Validation WMAE | `1,709.59` |

საინტერესოა, რომ საუკეთესო validation შედეგი "overfit" რეჟიმის კონფიგურაციამ მიიღო — ბევრი ხე, დიდი სიღრმე, პატარა learning rate. ეს XGBoost-ის მახასიათებელია: slow learning + large ensemble ხშირად validation-ზე კარგ შედეგს იძლევა, რადგან თითოეული ხე მცირე ნაბიჯს დებს და generalization-ი უკეთესია.

### XGBoost plots

ქვემოთ მოცემული plot აჩვენებს XGBoost runs-ის შედარებას validation WMAE-ის მიხედვით. საუკეთესო შედეგი `overfit_5` configuration-მა მიიღო.

<img src="notebooks/Tree%20Models/Plots/xgboost_sweep.png" alt="XGBoost sweep WMAE comparison" width="600">

Feature importance plot გვიჩვენებს, რომელ feature-ებს XGBoost ყველაზე მეტ გამოყენებას უკეთებდა.

<img src="notebooks/Tree%20Models/Plots/xgboost_feature_importance.png" alt="XGBoost feature importance" width="600">

შემდეგი plot აჩვენებს იმ Store-Dept წყვილებს, სადაც validation error ყველაზე მაღალი იყო. ყველაზე რთული სერიები აღმოჩნდა `(14, 92)`, `(14, 95)`, `(18, 92)`, `(10, 72)` და `(38, 38)`.

<img src="notebooks/Tree%20Models/Plots/xgboost_worst_store_dept.png" alt="XGBoost worst Store-Dept validation errors" width="600">

Holiday vs non-holiday error-იც რომ შევადაროთ:

- Non-holiday MAE: `1,699.73`
- Holiday MAE: `1,746.06`

DLinear-ისა და LightGBM-ისგან განსხვავებით, XGBoost-ს holiday კვირებზე ოდნავ მაღალი შეცდომა ჰქონდა, მაგრამ სხვაობა (`46.33`) მინიმალურია. ეს სავარაუდოდ `sample_weight=5` holiday weighting-ის ეფექტია — მოდელი holiday კვირებს train-ის დროს ახლოს ამუშავებს, მაგრამ spike-ების ზუსტი სიდიდე ყოველ წელს განსხვავდება.

### დასკვნა

XGBoost ამ პროექტში გამოვიყენეთ როგორც LightGBM-ის ალტერნატიული gradient boosting მოდელი, lag features-ის უფრო მდიდარი სეტით. საუკეთესო XGBoost configuration-მა (`overfit_5`) validation-ზე მიიღო `1,709.59` WMAE — ეს ყველა სხვა მოდელზე უკეთესი შედეგია: N-BEATS (`1,858.23`), LightGBM (`1,864.71`), TFT (`2,216.19`) და DLinear (`2,555.44`).

XGBoost-ის წარმატების მიზეზი ალბათ lag features-ის სიმდიდრეა: `lag_26` (ახლო ისტორია) და `lag_52` (სეზონური reference) rolling stats-თან ერთად კომბინაციაში ქმნის ძლიერ autoregressive signal-ს. ამ signal-ს slow-learning large ensemble კარგად ეუფლება, ამიტომ "overfit" კონფიგურაციაც კი validation-ზე კარგ შედეგს იძლევა.
