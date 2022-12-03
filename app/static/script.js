'use_strict';
var root = {
	data() 
	{
	return {
			classes: [],
			flash:[],
			result:[]
		}		
	},//data
	
	computed: {
		
		
	},//computed
	
	methods: {
		send_ws(data){
			let j = JSON.stringify(data);
			
			if (null == this.ws){
					this.ini_ws();
					setTimeout(()=>{
						this.ws.send(j);
					},1000);
			}
			else {
				
				setTimeout(()=>{
						console.log(j);
						this.ws.send(j);
					},500);
			}
		},//send_ws
		
		
		add_class(name, epoch, count, batch, lr){
			
				data = {
					action: 'fit',
					name: name,
					batch: batch,
					epoch: epoch,
					lr: lr, 
					imgs: [], 
					coef: 1,  
					load: count === false? 0 : count,  
				}
				this.send_ws(data);
			
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
			
	  	ini_ws(){
			let url = `ws://${location.hostname}:${location.port}/ws`;
			this.ws = new WebSocket(url);
			this.ws.onopen = (e) => 
			{
				console.log('ws open');
			}
				
			this.ws.onclose = (e) => 
			{
				this.ws = null;
			}
			this.ws.onmessage = (e) =>
			{	
				try {
				let data = JSON.parse(e.data);
				if ('predict' == data['action'])
					this.predict_handler(data);
				else if ('fit' == data['action'])
					this.fit_handler(e.data)
				} catch(e) {
					console.log(e);
				}
				console.log(e.data);
			}
			this.ws.onerror = (e) =>
			{
				console.log(e);
				this.ws.close();
			}
		},//ini_ws
		
		predict(name){
			this.send_ws({
				action: 'predict',
				name: name, 
			});
		},//predict
		
		predict_handler(data){
			console.log('res' in data, data)
			
			if ('res' in data){
				this.result.push(data);
				if (data.count == data.predicted)
					this.$refs.predbox.predicting = false;
			}
		},//predict
		
	},//methods
		
	mounted() {
		this.load_classes();
		this.ini_ws();
		
	},//mounted
	
	components: {
		'classes-box': Classes,
		'add-class': AddClass,
		'predict-box': Predict
	},//components
	
}//root

const app = Vue.createApp(root);
const vm = app.mount('#app');