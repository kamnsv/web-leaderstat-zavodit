'use_strict';
var root = {
	data() 
	{
	return {
			classes: [],
			flash:[]
		}		
	},//data
	
	computed: {
		
		
	},//computed
	
	methods: {
		add_class(){
			alert('')
		},//add_class
		
	    load_classes(){
		  
			(async () => {
				const rawResponse = await fetch('/classes', {
					method: 'GET',
				});
				try{
					this.classes = await rawResponse.json();
					for (cls of this.classes)
						cls.on = 0;
				} catch(e){
					this.flash.push('Ошибка сервиса api /classes')
				}
			})();
		
	  },//load_classes
		

	},//methods
		
	mounted() {
		this.load_classes();
		
	},//mounted
	
	components: {
		'classes-box': Classes,
		'add-class': AddClass
	},//components
	
}//root

const app = Vue.createApp(root);
const vm = app.mount('#app');