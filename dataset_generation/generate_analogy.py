import random
from transformers import set_seed
import json
import argparse

COUNTRY_CITY_MAP = {'Afghanistan': {'capital': 'Kabul', 'other_city': 'Kandahār'}, 'Albania': {'capital': 'Tirana', 'other_city': 'Elbasan'}, 'Algeria': {'capital': 'Algiers', 'other_city': 'Oran'}, 'Andorra': {'capital': 'Andorra la Vella', 'other_city': 'Escaldes-Engordany'}, 'Angola': {'capital': 'Luanda', 'other_city': 'Huambo'}, 'Argentina': {'capital': 'Buenos Aires', 'other_city': 'Córdoba'}, 'Armenia': {'capital': 'Yerevan', 'other_city': 'Gyumri'}, 'Australia': {'capital': 'Canberra', 'other_city': 'Sydney'}, 'Austria': {'capital': 'Vienna', 'other_city': 'Linz'}, 'Azerbaijan': {'capital': 'Baku', 'other_city': 'Gəncə'}, 'The Bahamas': {'capital': 'Nassau', 'other_city': 'Freeport City'}, 'Bangladesh': {'capital': 'Dhaka', 'other_city': 'Chittagong'}, 'Belarus': {'capital': 'Minsk', 'other_city': 'Homyel’'}, 'Belgium': {'capital': 'Brussels', 'other_city': 'Antwerp'}, 'Belize': {'capital': 'Belmopan', 'other_city': 'Belize City'}, 'Benin': {'capital': 'Cotonou', 'other_city': 'Porto-Novo'}, 'Bhutan': {'capital': 'Thimphu', 'other_city': 'Paro'}, 'Bolivia': {'capital': 'La Paz', 'other_city': 'Santa Cruz'}, 'Bosnia And Herzegovina': {'capital': 'Sarajevo', 'other_city': 'Banja Luka'}, 'Botswana': {'capital': 'Gaborone', 'other_city': 'Francistown'}, 'Brazil': {'capital': 'Brasília', 'other_city': 'São Paulo'}, 'Brunei': {'capital': 'Bandar Seri Begawan', 'other_city': 'Kuala Belait'}, 'Bulgaria': {'capital': 'Sofia', 'other_city': 'Plovdiv'}, 'Burkina Faso': {'capital': 'Ouagadougou', 'other_city': 'Bobo-Dioulasso'}, 'Burma': {'capital': 'Rangoon', 'other_city': 'Mandalay'}, 'Burundi': {'capital': 'Bujumbura', 'other_city': 'Gitega'}, 'Cabo Verde': {'capital': 'Praia', 'other_city': 'Mindelo'}, 'Cambodia': {'capital': 'Phnom Penh', 'other_city': 'Battambang'}, 'Cameroon': {'capital': 'Yaoundé', 'other_city': 'Douala'}, 'Canada': {'capital': 'Ottawa', 'other_city': 'Toronto'}, 'Central African Republic': {'capital': 'Bangui', 'other_city': 'Mbaïki'}, 'Chad': {'capital': 'N’Djamena', 'other_city': 'Bongor'}, 'Chile': {'capital': 'Santiago', 'other_city': 'Concepción'}, 'China': {'capital': 'Beijing', 'other_city': 'Shanghai'}, 'Colombia': {'capital': 'Bogotá', 'other_city': 'Medellín'}, 'Comoros': {'capital': 'Moroni', 'other_city': 'Fomboni'}, 'Congo (Brazzaville)': {'capital': 'Brazzaville', 'other_city': 'Pointe-Noire'}, 'Congo (Kinshasa)': {'capital': 'Kinshasa', 'other_city': 'Lubumbashi'}, 'Costa Rica': {'capital': 'San José', 'other_city': 'Alajuela'}, 'Croatia': {'capital': 'Zagreb', 'other_city': 'Split'}, 'Cuba': {'capital': 'Havana', 'other_city': 'Santiago de Cuba'}, 'Cyprus': {'capital': 'Nicosia', 'other_city': 'Limassol'}, 'Czechia': {'capital': 'Prague', 'other_city': 'Ostrava'}, 'Côte D’Ivoire': {'capital': 'Abidjan', 'other_city': 'Yamoussoukro'}, 'Denmark': {'capital': 'Copenhagen', 'other_city': 'Aarhus'}, 'Djibouti': {'capital': 'Djibouti', 'other_city': 'Ali Sabieh'}, 'Dominican Republic': {'capital': 'Santo Domingo', 'other_city': 'Santiago'}, 'Ecuador': {'capital': 'Quito', 'other_city': 'Guayaquil'}, 'Egypt': {'capital': 'Cairo', 'other_city': 'Alexandria'}, 'El Salvador': {'capital': 'San Salvador', 'other_city': 'Santa Ana'}, 'Equatorial Guinea': {'capital': 'Malabo', 'other_city': 'Bata'}, 'Eritrea': {'capital': 'Asmara', 'other_city': 'Mendefera'}, 'Estonia': {'capital': 'Tallinn', 'other_city': 'Tartu'}, 'Ethiopia': {'capital': 'Addis Ababa', 'other_city': 'Nazrēt'}, 'Fiji': {'capital': 'Suva', 'other_city': 'Lautoka'}, 'Finland': {'capital': 'Helsinki', 'other_city': 'Tampere'}, 'France': {'capital': 'Paris', 'other_city': 'Lyon'}, 'Gabon': {'capital': 'Libreville', 'other_city': 'Port-Gentil'}, 'The Gambia': {'capital': 'Banjul', 'other_city': 'Brikama'}, 'Georgia': {'capital': 'Tbilisi', 'other_city': 'Kutaisi'}, 'Germany': {'capital': 'Berlin', 'other_city': 'Stuttgart'}, 'Ghana': {'capital': 'Accra', 'other_city': 'Kumasi'}, 'Greece': {'capital': 'Athens', 'other_city': 'Thessaloníki'}, 'Guam': {'capital': 'Hagåtña', 'other_city': 'Maina'}, 'Guatemala': {'capital': 'Guatemala City', 'other_city': 'Quetzaltenango'}, 'Guinea': {'capital': 'Conakry', 'other_city': 'Guéckédou'}, 'Guinea-Bissau': {'capital': 'Bissau', 'other_city': 'Bafatá'}, 'Guyana': {'capital': 'Georgetown', 'other_city': 'New Amsterdam'}, 'Haiti': {'capital': 'Port-au-Prince', 'other_city': 'Cap-Haïtien'}, 'Honduras': {'capital': 'Tegucigalpa', 'other_city': 'San Pedro Sula'}, 'Hungary': {'capital': 'Budapest', 'other_city': 'Miskolc'}, 'Iceland': {'capital': 'Reykjavík', 'other_city': 'Akureyri'}, 'India': {'capital': 'New Delhi', 'other_city': 'Mumbai'}, 'Indonesia': {'capital': 'Jakarta', 'other_city': 'Surabaya'}, 'Iran': {'capital': 'Tehran', 'other_city': 'Mashhad'}, 'Iraq': {'capital': 'Baghdad', 'other_city': 'Mosul'}, 'Ireland': {'capital': 'Dublin', 'other_city': 'Cork'}, 'Israel': {'capital': 'Jerusalem', 'other_city': 'Tel Aviv-Yafo'}, 'Italy': {'capital': 'Rome', 'other_city': 'Milan'}, 'Jamaica': {'capital': 'Kingston', 'other_city': 'Spanish Town'}, 'Japan': {'capital': 'Tokyo', 'other_city': 'Ōsaka'}, 'Jordan': {'capital': 'Amman', 'other_city': 'Az Zarqā’'}, 'Kazakhstan': {'capital': 'Nur-Sultan', 'other_city': 'Almaty'}, 'Kenya': {'capital': 'Nairobi', 'other_city': 'Mombasa'}, 'North Korea': {'capital': 'Pyongyang', 'other_city': 'Namp’o'}, 'South Korea': {'capital': 'Seoul', 'other_city': 'Busan'}, 'Kosovo': {'capital': 'Pristina', 'other_city': 'Mamushë'}, 'Kuwait': {'capital': 'Kuwait City', 'other_city': 'Al Jahrā’'}, 'Kyrgyzstan': {'capital': 'Bishkek', 'other_city': 'Osh'}, 'Laos': {'capital': 'Vientiane', 'other_city': 'Louangphabang'}, 'Latvia': {'capital': 'Riga', 'other_city': 'Daugavpils'}, 'Lebanon': {'capital': 'Beirut', 'other_city': 'Tripoli'}, 'Lesotho': {'capital': 'Maseru', 'other_city': 'Mafeteng'}, 'Liberia': {'capital': 'Monrovia', 'other_city': 'Gbarnga'}, 'Libya': {'capital': 'Tripoli', 'other_city': 'Benghazi'}, 'Liechtenstein': {'capital': 'Vaduz', 'other_city': 'Mauren'}, 'Lithuania': {'capital': 'Vilnius', 'other_city': 'Kaunas'}, 'Luxembourg': {'capital': 'Luxembourg', 'other_city': 'Diekirch'}, 'Macedonia': {'capital': 'Skopje', 'other_city': 'Tetovo'}, 'Madagascar': {'capital': 'Antananarivo', 'other_city': 'Antsirabe'}, 'Malawi': {'capital': 'Lilongwe', 'other_city': 'Blantyre'}, 'Malaysia': {'capital': 'Kuala Lumpur', 'other_city': 'George Town'}, 'Maldives': {'capital': 'Male', 'other_city': 'Un’goofaaru'}, 'Mali': {'capital': 'Bamako', 'other_city': 'Sikasso'}, 'Malta': {'capital': 'Valletta', 'other_city': 'Sliema'}, 'Mauritania': {'capital': 'Nouakchott', 'other_city': 'Néma'}, 'Mauritius': {'capital': 'Port Louis', 'other_city': 'Curepipe'}, 'Mexico': {'capital': 'Mexico City', 'other_city': 'Guadalajara'}, 'Federated States Of Micronesia': {'capital': 'Palikir', 'other_city': 'Kolonia'}, 'Moldova': {'capital': 'Chisinau', 'other_city': 'Tiraspol'}, 'Mongolia': {'capital': 'Ulaanbaatar', 'other_city': 'Erdenet'}, 'Montenegro': {'capital': 'Podgorica', 'other_city': 'Tivat'}, 'Morocco': {'capital': 'Rabat', 'other_city': 'Casablanca'}, 'Mozambique': {'capital': 'Maputo', 'other_city': 'Matola'}, 'Namibia': {'capital': 'Windhoek', 'other_city': 'Rundu'}, 'Nepal': {'capital': 'Kathmandu', 'other_city': 'Jitpur'}, 'Netherlands': {'capital': 'The Hague', 'other_city': 'Amsterdam'}, 'New Zealand': {'capital': 'Wellington', 'other_city': 'Auckland'}, 'Nicaragua': {'capital': 'Managua', 'other_city': 'León'}, 'Niger': {'capital': 'Niamey', 'other_city': 'Maradi'}, 'Nigeria': {'capital': 'Abuja', 'other_city': 'Lagos'}, 'Norway': {'capital': 'Oslo', 'other_city': 'Bergen'}, 'Oman': {'capital': 'Muscat', 'other_city': 'As Sīb'}, 'Pakistan': {'capital': 'Islamabad', 'other_city': 'Karachi'}, 'Palau': {'capital': 'Ngerulmud', 'other_city': 'Koror'}, 'Panama': {'capital': 'Panama City', 'other_city': 'Colón'}, 'Papua New Guinea': {'capital': 'Port Moresby', 'other_city': 'Lae'}, 'Paraguay': {'capital': 'Asunción', 'other_city': 'San Lorenzo'}, 'Peru': {'capital': 'Lima', 'other_city': 'Callao'}, 'Philippines': {'capital': 'Manila', 'other_city': 'Quezon City'}, 'Poland': {'capital': 'Warsaw', 'other_city': 'Katowice'}, 'Portugal': {'capital': 'Lisbon', 'other_city': 'Porto'}, 'Qatar': {'capital': 'Doha', 'other_city': 'Al Wakrah'}, 'Romania': {'capital': 'Bucharest', 'other_city': 'Iaşi'}, 'Russia': {'capital': 'Moscow', 'other_city': 'Saint Petersburg'}, 'Rwanda': {'capital': 'Kigali', 'other_city': 'Nyanza'}, 'Samoa': {'capital': 'Apia', 'other_city': 'Afega'}, 'San Marino': {'capital': 'San Marino', 'other_city': 'Serravalle'}, 'Sao Tome And Principe': {'capital': 'São Tomé', 'other_city': 'Trindade'}, 'Saudi Arabia': {'capital': 'Riyadh', 'other_city': 'Jeddah'}, 'Senegal': {'capital': 'Dakar', 'other_city': 'Thiès'}, 'Serbia': {'capital': 'Belgrade', 'other_city': 'Niš'}, 'Sierra Leone': {'capital': 'Freetown', 'other_city': 'Bo'}, 'Slovakia': {'capital': 'Bratislava', 'other_city': 'Košice'}, 'Slovenia': {'capital': 'Ljubljana', 'other_city': 'Maribor'}, 'Solomon Islands': {'capital': 'Honiara', 'other_city': 'Gizo'}, 'Somalia': {'capital': 'Mogadishu', 'other_city': 'Kismaayo'}, 'South Africa': {'capital': 'Cape Town', 'other_city': 'Johannesburg'}, 'South Sudan': {'capital': 'Juba', 'other_city': 'Yei'}, 'Spain': {'capital': 'Madrid', 'other_city': 'Barcelona'}, 'Sri Lanka': {'capital': 'Colombo', 'other_city': 'Sri Jayewardenepura Kotte'}, 'Sudan': {'capital': 'Khartoum', 'other_city': 'Omdurman'}, 'Suriname': {'capital': 'Paramaribo', 'other_city': 'Nieuw Nickerie'}, 'Swaziland': {'capital': 'Mbabane', 'other_city': 'Lobamba'}, 'Sweden': {'capital': 'Stockholm', 'other_city': 'Göteborg'}, 'Switzerland': {'capital': 'Bern', 'other_city': 'Geneva'}, 'Syria': {'capital': 'Damascus', 'other_city': 'Aleppo'}, 'Taiwan': {'capital': 'Taipei', 'other_city': 'Kaohsiung'}, 'Tajikistan': {'capital': 'Dushanbe', 'other_city': 'Khŭjand'}, 'Tanzania': {'capital': 'Dar es Salaam', 'other_city': 'Dodoma'}, 'Thailand': {'capital': 'Bangkok', 'other_city': 'Chiang Mai'}, 'Timor-Leste': {'capital': 'Dili', 'other_city': 'Baucau'}, 'Togo': {'capital': 'Lomé', 'other_city': 'Sokodé'}, 'Tonga': {'capital': 'Nuku‘alofa', 'other_city': 'Neiafu'}, 'Trinidad And Tobago': {'capital': 'Port of Spain', 'other_city': 'San Fernando'}, 'Tunisia': {'capital': 'Tunis', 'other_city': 'Sfax'}, 'Turkey': {'capital': 'Ankara', 'other_city': 'Istanbul'}, 'Turkmenistan': {'capital': 'Ashgabat', 'other_city': 'Türkmenabat'}, 'Uganda': {'capital': 'Kampala', 'other_city': 'Mbale'}, 'Ukraine': {'capital': 'Kyiv', 'other_city': 'Kharkiv'}, 'United Arab Emirates': {'capital': 'Abu Dhabi', 'other_city': 'Dubai'}, 'United Kingdom': {'capital': 'London', 'other_city': 'Birmingham'}, 'United States': {'capital': 'Washington', 'other_city': 'New York'}, 'Uruguay': {'capital': 'Montevideo', 'other_city': 'Rivera'}, 'Uzbekistan': {'capital': 'Tashkent', 'other_city': 'Farg‘ona'}, 'Vanuatu': {'capital': 'Port-Vila', 'other_city': 'Luganville'}, 'Venezuela': {'capital': 'Caracas', 'other_city': 'Maracaibo'}, 'Vietnam': {'capital': 'Hanoi', 'other_city': 'Ho Chi Minh City'}, 'Yemen': {'capital': 'Sanaa', 'other_city': 'Aden'}, 'Zambia': {'capital': 'Lusaka', 'other_city': 'Kitwe'}, 'Zimbabwe': {'capital': 'Harare', 'other_city': 'Bulawayo'}}


def generate_edits(edit_out):
    editing_dataset = []
    country_list = list(COUNTRY_CITY_MAP.keys())
    change_capital_list = random.sample(country_list, len(country_list)//2)
    keep_capital_list = list(set(country_list).difference(set(change_capital_list)))

    for country in change_capital_list:
        curr_capital = COUNTRY_CITY_MAP[country]['capital']
        new_capital = COUNTRY_CITY_MAP[country]['other_city']

        item = {'subject': country, 'true_target': curr_capital,
                 'target_1': new_capital, 'target_2': curr_capital, 'prompt': f'The capital of {country} is '}
        editing_dataset.append(item)

        # new edit for keeping old capital as city
        item = {'subject': curr_capital, 'true_target': country,
                 'target_1': country, 'target_2': country, 'prompt': f'{curr_capital} is a city in '}
        editing_dataset.append(item)
        
    # in case model is not familiar with existing (capital, country) pair
    #for country in keep_capital_list:
    #    curr_capital = COUNTRY_CITY_MAP[country]['capital']
    #    item = {'subject': country, 'true_target': curr_capital,
    #             'target_1': curr_capital, 'target_2': curr_capital, 'prompt': f'The capital of {country} is '}
    #    editing_dataset.append(item)

    with open(edit_out, 'w') as f:
        json.dump(editing_dataset, f, indent=4)
    
    return change_capital_list, keep_capital_list

def generate_dataset(num_samples, dataset_out, change_capital_list, keep_capital_list):
    dataset = []
    country_list = change_capital_list + keep_capital_list
    all_pairs = [(c1, c2) for c1 in change_capital_list for c2 in keep_capital_list]
    num_samples = min(num_samples, len(all_pairs))

    random.shuffle(all_pairs)
    for c1, c2 in all_pairs:
            c1_capital, c2_capital = COUNTRY_CITY_MAP[c1]['capital'], COUNTRY_CITY_MAP[c2]['capital']
            question = f"{c1_capital} is to {c1} like {c2_capital} is to _."
            gold_answer = c2
            other_choice = random.sample([c for c in country_list if c != gold_answer], 1)[0]
            choices = [gold_answer, other_choice]
            random.shuffle(choices)
            synth_expl_1 = f"{c2_capital} is a city in {c2} as {c1_capital} is a city in {c1}."
            synth_expl_2 = f"The capital of {c2} is {c2_capital} as the capital of {c1} is {c1_capital}."

            # generate only related edits
            curr_capital = COUNTRY_CITY_MAP[c1]['capital']
            new_capital = COUNTRY_CITY_MAP[c1]['other_city']
            edit_1 = {'subject': c1, 'true_target': c1_capital, 'target_1': new_capital, 'target_2': curr_capital, 'prompt': f'The capital of {c1} is '}
            edit_2 = {'subject': curr_capital, 'true_target': c1, 'target_1': c1, 'target_2': c1, 'prompt': f'{curr_capital} is a city in '}

            dataset.append({"question": question, "choice_A": choices[0],
            "choice_B": choices[1], "label_txt": gold_answer, "label": ['A', 'B'][choices.index(gold_answer)],
            "synthetic_explanation_1": synth_expl_1,
            "synthetic_explanation_2": synth_expl_2,
            "related_edits": [edit_1, edit_2]})

    with open(dataset_out, 'w') as f:
        json.dump(dataset[:num_samples], f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset for model editing and factual evaluation')
    parser.add_argument('--editing-data', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=False, default=123)

    args = parser.parse_args()

    set_seed(args.seed)
    change_capital_list, keep_capital_list = generate_edits(args.editing_data)
    generate_dataset(args.size, args.eval_data, change_capital_list, keep_capital_list)