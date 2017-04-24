print("database2 is loaded")

def start_db(db_name, _echo=False, _DB_RESET=False):
	import numpy as np
	import os, json
	from hashlib import md5

	import sqlalchemy as sa
	from sqlalchemy import create_engine, MetaData
	from sqlalchemy.orm import scoped_session, sessionmaker
	from sqlalchemy.ext.declarative import declarative_base
	from sqlalchemy.types import TypeDecorator, LargeBinary, String	

	from sqlalchemy import Table, Column, ForeignKey
	from sqlalchemy.types import Integer, String, Float, Boolean
	from sqlalchemy.orm import mapper, relation

	db_path = "E:/gnuplot/notebook/db/"+db_name+".db"
	if _DB_RESET:
	    if os.path.exists(db_path):os.remove(db_path)
	engine = create_engine("sqlite:///"+db_path, echo=_echo)
	# engine = create_engine("sqlite:///:memory:", echo=True)

	session = scoped_session(sessionmaker(bind = engine))
	Base = declarative_base()
	metadata = MetaData()

	#### New SQLAlchemy-Type #####################
	class Npyz (TypeDecorator):    
		impl = String
	#     def __init__(self, *args, **kwargs):
	#         TypeDecorator.__init__(self, *args, **kwargs)
	#         self.path = None
	#         print("now initializing", self.path[-10:] if self.path is not None else "path is None")
			
		def process_bind_param(self, value, dialect):
			import os
			_path = os.sep.join(["E:","gnuplot", "notebook", "db", "data", ""]) +\
						md5(value.tostring()).hexdigest()+".npz"
			if not os.path.exists(_path): np.savez_compressed(_path, value)
			return _path

		def process_result_value(self, value, dialect):
			return value

	class ArrayType(TypeDecorator):
		impl = String
		def process_bind_param(self, value, dialect): return json.dumps(value)
		def process_result_value(self, value, dialect): return json.loads(value)
		def copy(self): return ArrayType(self.impl.length)
	##############################################


	from sqlalchemy import Table, Column, ForeignKey
	from sqlalchemy.types import Integer, String, Float, Boolean
	from sqlalchemy.orm import mapper, relation

	class Experimenter(Base): # 1.
		__tablename__="experimenter"
		id = Column(Integer, primary_key=True)
		full_name = Column(String, unique=True)
		email_adress = Column(String) 
		affiliation = Column(ArrayType())
		f_name = Column(String)
		l_name = Column(String)

		def __repr__(self):
			return "{}: {}\n{}".format(self.full_name, self.email_adress, self.affiliation)

	note_proto_table = Table('association', Base.metadata,
		Column('experiment_note_id', Integer, ForeignKey('experiment_note.id')),
		Column('experiment_protocol_id', Integer, ForeignKey('experiment_protocol.id')) )

	class Experiment_protocol(Base): # 2.
		__tablename__="experiment_protocol"
		id = Column(Integer, primary_key=True)
		name = Column(String) 
		writer = Column(String)
		reference = Column(ArrayType()) # TODO: make it mendeley object 
		body = Column(String) # TODO: make it sphinx object
		variable_name = Column(String) # TODO: make it great
		notes = relation(
			"Experiment_note",
			secondary=note_proto_table,
			back_populates="protocols")

	class Experiment_note(Base): # 3.
		__tablename__="experiment_note"
		id = Column(Integer, primary_key=True)
		variables = Column(String)
		notes = Column(String)
		experimenter_id = Column(Integer, ForeignKey("experimenter.id"))
		protocols = relation(
			"Experiment_protocol",
			secondary=note_proto_table,
			back_populates="notes")
		lasertrap_rawdatas = relation("Lasertrap_rawdata", backref="experiment_note")
		
	class Construct(Base): # 4.1
		__tablename__="construct"
		id = Column(Integer, primary_key=True)
		name = Column(String)
		sequence = Column(String)
		reference = Column(ArrayType(), nullable=True)
		lasertrap_rawdatas = relation("Lasertrap_rawdata", backref="construct")

	class Construct2(Base): # 4.2
		__tablename__="construct2"
		id = Column(Integer, primary_key=True)
		name = Column(String)
		sequence = Column(String)
		reference = Column(ArrayType(), nullable=True)
		lasertrap_rawdatas = relation("Lasertrap_rawdata", backref="construct2")

	class Lasertrap_rawdata(Base): # 5.
		__tablename__="lasertrap_rawdata"
		id = Column(Integer, primary_key=True)
		date = Column(String, nullable=False)
		dilution_rate = Column(String, nullable=False)
		dilution_rate2 = Column(String, nullable=True)
		beads_diameter = Column(Float, nullable=False)
		beads_name = Column(String, nullable=False)
		raw_t_path = Column(Npyz, nullable=False)
		raw_x_path = Column(Npyz, nullable=False)
		raw_y_path = Column(Npyz, nullable=False)
	#     raw_I_path = Column(Npyz, nullable=False)
		length = Column(Integer, nullable=False)
		motile_fraction = Column(Float, nullable=False)
		sampling_rate = Column(Float, default=20*1000)
		assay_buffer = Column(String)
		ATP_conc = Column(Float, default=2) # mM
		casein_conc = Column(Float, default=4) # mg/ml
		glycerol_conc = Column(Float, nullable=False)
		original_path = Column(String, nullable=True)
		stiffness = Column(Float, nullable=False)
		file_number = Column(Integer)
		beads_number = Column(Integer)
		beads_separate_number = Column(Integer)

		expnote_id = Column(Integer, ForeignKey("experiment_note.id"))
		construct_id = Column(Integer, ForeignKey("construct.id"))
		construct2_id = Column(Integer, ForeignKey("construct2.id"))
		lasertrap_analyses = relation("Lasertrap_analysis", backref="lasertrap_rawdata")

		def __repr__(self):
			import numpy as np
			from pprint import pprint
			attrs = { attr:getattr(self, attr) for attr in dir(self) if not attr.startswith("_") }
			if isinstance(attrs["raw_t_path"], np.ndarray):
				for k in ["raw_t_path", "raw_x_path", "raw_y_path"]:
					attrs[k] = os.sep.join(["E:","gnuplot", "notebook", "db", "data", ""]) +\
						md5(attrs[k].tostring()).hexdigest()+".npz"
			return """\n\
date : {date}
construct : {construct_name}
dilution_rate : {dilution_rate} x
construct2 : {construct_name2}
dilution_rate2 : {dilution_rate2} x
data_number : {file_number}-{beads_number}-{beads_separate_number}
assay buffer : {assay_buffer}, {ATP_conc}mM ATP, {casein_conc}mg/ml Casein, {glycerol_conc}%, GCOβ, pH 6.8(KOH)
beads : {beads_name}, {beads_diameter} nm, 0.1 nM
length : {min} min {sec:0.4} sec
stiffness : {stiffness:0.5} pN/nm (sd={sd:0.5})
protein concentration : TODO
raw data paths -> t:{raw_t_path}, x:{raw_x_path}, y:{raw_y_path}
original_path : {original_path}
""".format( min=self.length/20/1000//60, sec=self.length/20/1000%60,
			  sd=1./np.sqrt(self.stiffness/4.02), construct_name=self.construct.name, 
			  construct_name2=self.construct2.name if self.construct2 is not None else None,
			  **attrs )

	class Analysis_protocol(Base): # 6.
		__tablename__="analysis_protocol"
		id = Column(Integer, primary_key=True)
		name = Column(String)
		writer = Column(String)
		reference = Column(ArrayType())
		body = Column(String)
		variables_name = Column(String)

	class Lasertrap_analysis(Base): # 7.
		__tablename__="lasertrap_analysis"
		id = Column(Integer, primary_key=True)
		name = Column(String)
		log = Column(String)
		func_log = Column(String)
		lasertrap_rawdata_id = Column(Integer, ForeignKey("lasertrap_rawdata.id"))
		runs = relation("Run", backref="lasertrap_analysis")

	class Run(Base): #8.
		__tablename__="run"
		id = Column(Integer, primary_key=True)
		start_i = Column(Integer)
		end_i = Column(Integer)
		velocity_max = Column(Float)
		force_max = Column(Float)
		duration_time = Column(Float)
		isStall = Column(Boolean)
		lasertrap_analysis_id = Column(Integer, ForeignKey("lasertrap_analysis.id"))

	Base.metadata.create_all(engine)

	return {"engine" : engine, 
			"session" : session, 
			"Experimenter" : Experimenter, 
			"Experiment_protocol" : Experiment_protocol, 
			"Experiment_note" : Experiment_note, 
			"Construct" : Construct, 
			"Lasertrap_rawdata" : Lasertrap_rawdata, 
			"Analysis_protocol" : Analysis_protocol, 
			"Lasertrap_analysis" : Lasertrap_analysis, 
			"Run" : Run, 
			"Construct2": Construct2}

if __name__ == "__main__":
	db_d=start_db("data_test", _echo=True, _DB_RESET=False)
	import pandas as pd

	session, Construct, engine = db_d["session"], db_d["Construct"], db_d["engine"]
	import pandas as pd
	cn=[]#本来ここに名前のリストが入る
	session.add_all([Construct(name=cn) for cn in cnams])
	print(session)
	session.commit()

